import os
import sys
import time
import copy
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import backward
from torchvision import transforms, utils
from scipy import linalg
from tqdm import tqdm

import misc
from dataset import get_dataset, ResamplingDataset
from .calc_inception import load_patched_inception_v3


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def load_condition_sample(sample_dir, batch_size):
    IMG_EXTS = ['jpg', 'png', 'jpeg']
    samples = []
    for ext in IMG_EXTS:
        samples.extend(list(Path(sample_dir).glob(f'*.{ext}')))
    assert len(samples) >= batch_size, f"Need more samples. {len(samples)} < {batch_size}"
    cond_samples = []

    trf = [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [5], inplace=True),
    ]
    transform = transforms.Compose(trf)

    for i, p in enumerate():
        if batch_size != -1 and i == batch_size:
            break
        cond_samples.append(transform(io.imread(p)[..., None])[None, ...])
    return torch.cat(cond_samples, dim=0)


class FIDTracker():
    def __init__(self, cfg, rank, num_gpus, out_dir, use_tqdm=False):
        print(f"[{os.getpid()}] rank: {rank}({num_gpus} gpus)")
        fid_cfg = cfg.EVAL.FID
        inception_path = fid_cfg.INCEPTION_CACHE

        self.num_gpus = num_gpus
        self.rank = rank
        self.device = torch.device('cuda', rank) if num_gpus > 1 else 'cuda'
        self.cfg = fid_cfg
        self.log = logging.getLogger(f'GPU{rank}')
        self.out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        self.real_mean = []  # ndarray(n_class, 2048) float32
        self.real_cov = []  # ndarray(n_class, 2048, 2048) float64
        self.idx_to_class = []
        self.k_iters = []
        self.fids = []
        self.cond_samples = None
        self.latent_size = cfg.MODEL.LATENT_SIZE
        self.conditional = True if cfg.N_CLASSES > 1 else False
        self.num_classes = cfg.N_CLASSES
        self.model_bs = 8
        self.use_tqdm = use_tqdm
        if fid_cfg.SAMPLE_DIR:
            self.cond_samples = load_condition_sample(fid_cfg.SAMPLE_DIR, self.model_bs)
            self.cond_samples = self.cond_samples.to(self.device)
            self.log.info(f"using smaple directory: {fid_cfg.SAMPLE_DIR}."
                          f"Get {self.cond_samples.shape[0]} conditional sample")
            self.model_bs = self.cond_samples.shape[0]

        # get inception V3 model
        start = time.time()
        self.log.info("load inception model...")
        if num_gpus == 1:
            self.inceptionV3 = load_patched_inception_v3().eval().to(self.device)
        else:
            if self.rank != 0:
                torch.distributed.barrier()
            self.inceptionV3 = load_patched_inception_v3().eval().to(self.device)
            if self.rank == 0:
                torch.distributed.barrier()
        self.log.info("load inception model complete ({:.2f})".format(time.time() - start))

        # get features for real images
        if inception_path:
            self.log.info("load inception from cache file")
            with open(inception_path, 'rb') as f:
                embeds = pickle.load(f)
                self.real_mean = embeds['mean']
                self.real_cov = embeds['cov']
                self.idx_to_class = embeds['idx_to_class']
        else:
            self.extract_feature_from_real_images(cfg)
            if self.rank == 0:
                self.log.info(f"save inception cache in {self.out_dir}")
                with open(self.out_dir / 'inception_cache.pkl', 'wb') as f:
                    pickle.dump(dict(mean=self.real_mean, cov=self.real_cov, idx_to_class=self.idx_to_class), f)

        self.val_dataset = ResamplingDataset(cfg.DATASET, cfg.RESOLUTION)
        self.log.info(f"validation data samples: {len(self.val_dataset)}")

    def calc_fid(self, generator, k_iter, save=False, eps=1e-6):
        self.log.info(f'get fid on {k_iter * 1000} iterations')
        start = time.time()
        sample_features = self.extract_feature_from_model(generator)

        fids = []
        for i, sample_feature in enumerate(sample_features):
            sample_mean = np.mean(sample_feature, 0)
            sample_cov = np.cov(sample_feature, rowvar=False)

            cov_sqrt, _ = linalg.sqrtm(sample_cov @ self.real_cov[i], disp=False)

            if not np.isfinite(cov_sqrt).all():
                self.log.warning('product of cov matrices is singular')
                offset = np.eye(sample_cov.shape[0]) * eps
                cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (self.real_cov[i] + offset))

            if np.iscomplexobj(cov_sqrt):
                if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                    m = np.max(np.abs(cov_sqrt.imag))

                    raise ValueError(f'Imaginary component {m}')

                cov_sqrt = cov_sqrt.real

            mean_diff = sample_mean - self.real_mean[i]
            mean_norm = mean_diff @ mean_diff

            trace = np.trace(sample_cov) + np.trace(self.real_cov[i]) - 2 * np.trace(cov_sqrt)
            fids.append(mean_norm + trace)

        finish = time.time()
        total_time = finish - start
        self.log.info(f'FID in {str(1000 * k_iter).zfill(6)} iterations: "{fids}". [costs {round(total_time, 2)} sec(s)]')
        self.k_iters.append(k_iter)
        self.fids.append(fids)

        if self.rank == 0 and save:
            with open(self.out_dir / 'fid.txt', 'a+') as f:
                f.write(f'{k_iter}: {fids}\n')
            result_dict = {"results": {"fid50k_full": fids[0]},
                           "metric": "fid50k_full",
                           "total_time": total_time,
                           "total_time_str": f"{int(total_time // 60)}m {int(total_time % 60)}s",
                           "num_gpus": self.num_gpus,
                           "snapshot_pkl": "none",
                           "timestamp": time.time()}
            with open(self.out_dir / 'metric-fid50k_full.jsonl', 'at') as f:
                f.write(json.dumps(result_dict) + '\n')

        return fids

    @torch.no_grad()
    def extract_feature_from_real_images(self, cfg):
        for i in range(self.num_classes):
            start = time.time()
            self.idx_to_class.append(None)  # no class name now
            self.log.info(f'extract features from real "{self.idx_to_class[i]}" images...')
            dataset = get_dataset(cfg.DATASET, cfg.RESOLUTION)
            num_items = len(dataset)
            num_items = min(num_items, self.cfg.N_SAMPLE)

            item_subset = [(i * self.num_gpus + self.rank) % num_items
                           for i in range((num_items - 1) // self.num_gpus + 1)]
            self.log.info(f"#item subset: {len(item_subset)}")

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=item_subset, batch_size=4, num_workers=2)

            features = []
            loader = iter(dataloader)
            n_batch = len(item_subset) // self.cfg.BATCH_SIZE
            if len(item_subset) % self.cfg.BATCH_SIZE != 0:
                n_batch += 1
            progress_bar = range(n_batch)
            if self.use_tqdm:
                progress_bar = tqdm(progress_bar)

            for _ in progress_bar:
                imgs, *_ = next(loader)
                imgs = imgs.to(self.device)
                feature = self.inceptionV3(imgs)[0].view(imgs.shape[0], -1)
                if self.num_gpus > 1:
                    _features = []
                    for src in range(self.num_gpus):
                        y = feature.clone()
                        torch.distributed.broadcast(y, src=src)
                        _features.append(y)
                    feature = torch.stack(_features, dim=1).flatten(0, 1)
                    self.log.debug(f"feature: {feature.shape}")
                features.append(feature)

            features = torch.cat(features, 0).cpu().numpy()
            self.real_mean.append(np.mean(features, 0))
            self.real_cov.append(np.cov(features, rowvar=False))

            self.log.info(f"complete({round(time.time() - start, 2)} secs)."
                          f"total extracted features: {features.shape[0]}")

    @torch.no_grad()
    def extract_feature_from_model(self, generator):
        num_sample = self.cfg.N_SAMPLE // self.num_gpus
        n_batch = num_sample // self.model_bs
        resid = num_sample % self.model_bs
        features_list = []

        for class_idx in range(self.num_classes):
            loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=self.model_bs,
                                                 shuffle=False,
                                                 num_workers=2)
            loader = sample_data(loader)
            features = []
            idx_iterator = range(n_batch + 1)
            if self.use_tqdm:
                idx_iterator = tqdm(idx_iterator)

            for i in idx_iterator:
                batch = resid if i == n_batch else self.model_bs
                if batch == 0:
                    continue

                body_imgs, face_imgs, mask = [x[:batch].to(self.device) for x in next(loader)]
                masked_body = torch.cat(((body_imgs * mask), mask), dim=1)
                latent = torch.randn(batch, self.latent_size, device=self.device)
                fake_label = torch.LongTensor([class_idx] * batch).to(self.device)

                cond_samples = None
                if self.cond_samples is not None:
                    cond_samples = self.cond_samples[:batch]

                imgs, _ = generator([latent], labels_in=fake_label, style_in=face_imgs, content_in=masked_body)
                # if self.rank == 0:
                #     utils.save_image(
                #         imgs,
                #         self.out_dir / f'fake-{i}.png',
                #         nrow=4,
                #         normalize=True,
                #         range=(-1, 1),
                #     )

                feature = self.inceptionV3(imgs[:, :3, :, :])[0].view(imgs.shape[0], -1)
                if self.num_gpus > 1:
                    _features = []
                    for src in range(self.num_gpus):
                        y = feature.clone()
                        torch.distributed.broadcast(y, src=src)
                        _features.append(y)
                    feature = torch.stack(_features, dim=1).flatten(0, 1)
                    self.log.debug(f"feature: {feature.shape}")
                features.append(feature)

            features_list.append(torch.cat(features, 0).cpu().numpy())
        return features_list

    def plot_fid(self):
        self.log.info(f"save FID figure in {str(self.out_dir / 'fid.png')}")

        self.fids = np.array(self.fids).T
        plt.xlabel('k iterations')
        plt.ylabel('FID')
        for fids in self.fids:
            plt.plot(self.k_iters, fids)
        plt.legend([self.idx_to_class[idx] for idx in range(self.num_classes)], loc='upper right')
        plt.savefig(self.out_dir / 'fid.png')


def subprocess_fn(rank, args, cfg, temp_dir):
    from models import Generator
    args.local_rank = rank
    if args.num_gpus > 1:
        torch.cuda.set_device(rank)
        init_method = f"file://{os.path.join(temp_dir, '.torch_distributed_init')}"
        torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)
    misc.create_logger(**vars(args))
    device = torch.device('cuda', rank) if args.num_gpus > 1 else 'cuda'
    fid_tracker = FIDTracker(cfg, rank, args.num_gpus, args.out_dir, use_tqdm=True if rank == 0 else False)

    # for ckpt in ckpts:
    if rank == 0:
        print("create output dirtectory if not exists")
        Path(args.out_dir).mkdir(exist_ok=True, parents=True)
        print(f"calculating fid of {str(args.ckpt)}")

    for ckpt in [args.ckpt]:
        k_iter = int(str(ckpt.name)[5:11]) / 1e3
        ckpt = torch.load(ckpt)

        # latent_dim, label_size, resolution
        assert cfg.N_CLASSES >= 1, f"#classes must greater than 0"
        label_size = 0 if cfg.N_CLASSES == 1 else cfg.N_CLASSES
        g = Generator(cfg.MODEL.LATENT_SIZE,
                      label_size,
                      cfg.RESOLUTION,
                      embedding_size=cfg.MODEL.EMBEDDING_SIZE,
                      extra_channels=cfg.MODEL.EXTRA_CHANNEL,
                      dlatents_size=256,
                      is_training=False)
        g.load_state_dict(ckpt['g_ema'])
        g = copy.deepcopy(g).eval().requires_grad_(False).to(device)

        if args.num_gpus > 1:
            torch.distributed.barrier()
        fid_tracker.calc_fid(g, k_iter, save=True)

    if rank == 0:
        fid_tracker.plot_fid()


if __name__ == '__main__':
    import argparse
    import tempfile
    from config import get_cfg_defaults

    parser = argparse.ArgumentParser()
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument("--cfg", required=True, help="path to the configuration file")
    parser.add_argument("--gpus", type=int, default=1, dest='num_gpus')
    parser.add_argument('--out_dir', type=str, default='/tmp/fid_result')
    parser.add_argument('--ckpt', type=str, required=True, metavar='CHECKPOINT', help='model ckpt or dir')
    parser.add_argument("--debug", action='store_true', default=False, help="whether to use debug mode")

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)

    args.ckpt = Path(args.ckpt)
    ckpts = sorted(list(args.ckpt.glob('*.pt'))) if args.ckpt.is_dir() else [args.ckpt]

    assert args.num_gpus >= 1
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, cfg=cfg, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, cfg, temp_dir), nprocs=args.num_gpus)

