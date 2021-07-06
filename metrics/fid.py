import os
import time
import copy
import json
import pickle
import logging
from pathlib import Path

import torch
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm
from torch import nn
from torchvision import transforms, utils

import misc
from dataset import get_dataset, ResamplingDatasetV2
from .calc_inception import load_patched_inception_v3


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class FIDTracker():
    def __init__(self, cfg, rank, num_gpus, out_dir, use_tqdm=False):
        print(f"[{os.getpid()}] rank: {rank}({num_gpus} gpus)")
        fid_cfg = cfg.EVAL.FID
        inception_path = fid_cfg.inception_cache

        self.num_gpus = num_gpus
        self.rank = rank
        self.device = torch.device('cuda', rank) if num_gpus > 1 else 'cuda'
        self.cfg = fid_cfg
        self.log = logging.getLogger(f'GPU{rank}')
        self.out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        self.real_means = None  # [ndarray(2048) float32] * num_class
        self.real_covs = None  # [ndarray(2048, 2048) float64] * num_class
        self.idx_to_class = []
        self.k_iters = []
        self.fids = []
        self.latent_size = cfg.MODEL.z_dim
        self.conditional = True if cfg.num_classes > 1 else False
        self.num_classes = cfg.num_classes
        self.model_bs = 8
        self.use_tqdm = use_tqdm

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
                self.real_means = embeds['mean']
                self.real_covs = embeds['cov']
                self.idx_to_class = embeds['idx_to_class']
        else:
            self.real_means, self.real_covs = self.extract_feature_from_images(cfg)
            if self.rank == 0:
                self.log.info(f"save inception cache in {self.out_dir}")
                with open(self.out_dir / 'inception_cache.pkl', 'wb') as f:
                    pickle.dump(dict(mean=self.real_means, cov=self.real_covs, idx_to_class=self.idx_to_class), f)

        # self.val_dataset = ResamplingDataset(cfg.DATASET, cfg.resolution)
        self.val_dataset = ResamplingDatasetV2(cfg.DATASET, 256, split='val')
        # self.val_dataset = get_dataset(cfg.DATASET, cfg.resolution, split='val')
        self.log.info(f"validation data samples: {len(self.val_dataset)}")
    
    def calc_fid(self, candidate, k_iter=0, save=False, eps=1e-6):
        assert self.real_means is not None and self.real_covs is not None
        start = time.time()
        
        # single dispatch
        if isinstance(candidate, torch.nn.Module):
            self.log.info(f"get FID on {k_iter * 1000} iteration") 
            sample_means, sample_covs = self.extract_feature_from_model(candidate)
        else:
            self.log.info(f"get FID on {candidate.DATASET.roots[0]} iteration")
            sample_means, sample_covs = self.extract_feature_from_images(candidate)

        assert len(sample_means) == len(self.real_means) == len(sample_covs) == len(self.real_covs)
        fids = []
        for i, (sample_mean, sample_cov) in enumerate(zip(sample_means, sample_covs)):
            self.log.debug(f"mean: {sample_mean.shape}, cov: {sample_cov.shape}")
            cov_sqrt, _ = linalg.sqrtm(sample_cov @ self.real_covs[i], disp=False)
            self.log.info("sqrtm done")
            if not np.isfinite(cov_sqrt).all():
                self.log.warning('product of cov matrices is singular')
                offset = np.eye(sample_cov.shape[0]) * eps
                cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (self.real_covs[i] + offset))

            if np.iscomplexobj(cov_sqrt):
                if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                    m = np.max(np.abs(cov_sqrt.imag))

                    raise ValueError(f'Imaginary component {m}')

                cov_sqrt = cov_sqrt.real

            mean_diff = sample_mean - self.real_means[i]
            mean_norm = mean_diff @ mean_diff

            trace = np.trace(sample_cov) + np.trace(self.real_covs[i]) - 2 * np.trace(cov_sqrt)
            fids.append(mean_norm + trace)

        total_time = time.time() - start
        self.log.info(f'FID on {1000 * k_iter} iterations: "{fids}". [costs {total_time: .2f} sec(s)]')
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
    def extract_feature_from_images(self, cfg):
        means, covs = [], []
        for i in range(self.num_classes):
            start = time.time()
            self.idx_to_class.append(None)  # no class name now
            self.log.info(f'extract features from real "{self.idx_to_class[i]}" images...')
            dataset = get_dataset(cfg.DATASET, cfg.resolution, split='all')
            num_items = len(dataset)
            num_items = min(num_items, self.cfg.n_sample)
            self.log.info(f"total: {num_items} real images")

            item_subset = [(i * self.num_gpus + self.rank) % num_items
                           for i in range((num_items - 1) // self.num_gpus + 1)]
            self.log.info(f"#item subset: {len(item_subset)}")

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=item_subset, batch_size=4, num_workers=2)

            features = []
            loader = iter(dataloader)
            n_batch = len(item_subset) // self.cfg.batch_size
            if len(item_subset) % self.cfg.batch_size != 0:
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
            means.append(np.mean(features, 0))
            covs.append(np.cov(features, rowvar=False))

            self.log.info(f"complete({round(time.time() - start, 2)} secs)."
                          f"total extracted features: {features.shape[0]}")
        return means, covs

    @torch.no_grad()
    def extract_feature_from_model(self, generator, k_iter):
        
        sample_means, sample_covs = [], []
        num_sample = self.cfg.n_sample // self.num_gpus
        n_batch = num_sample // self.model_bs
        resid = num_sample % self.model_bs
        num_items = len(self.val_dataset)
        item_subset = [(i * self.num_gpus + self.rank) % num_items
                       for i in range((num_items - 1) // self.num_gpus + 1)]

        for class_idx in range(self.num_classes):
            loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=self.model_bs,
                                                 shuffle=False,
                                                 sampler=item_subset,
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

                imgs, _ = generator([latent], labels_in=fake_label, style_in=face_imgs, content_in=masked_body, noise_mode='const')
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

            features = torch.cat(features, 0).cpu().numpy()
            sample_means.append(np.mean(features, 0))
            sample_covs.append(np.cov(features, rowvar=False))

        return sample_means, sample_covs

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
        
    if args.real:
        fid_tracker.calc_fid(cfg)
        return

    args.ckpt = Path(args.ckpt)
    args.ckpt = sorted(args.ckpt.glob('*.pt')) if args.ckpt.is_dir() else [args.ckpt]
    for ckpt in args.ckpt:
        print(f"calculating fid of {str(ckpt)}")
        k_iter = int(str(ckpt.name)[5:11]) / 1e3
        ckpt = torch.load(ckpt)

        # latent_dim, label_size, resolution
        assert cfg.num_classes >= 1, f"#classes must greater than 0"
        label_size = 0 if cfg.num_classes == 1 else cfg.num_classes
        g = Generator(
            cfg.MODEL.z_dim,
            label_size,
            cfg.resolution,
            extra_channels=cfg.MODEL.extra_channel,
            use_style_encoder=cfg.MODEL.use_style_encoder,
            use_content_encoder=cfg.MODEL.use_content_encoder,
            map_kwargs=cfg.MODEL.G_MAP,
            style_encoder_kwargs=cfg.MODEL.STYLE_ENCODER,
            synthesis_kwargs=cfg.MODEL.G_SYNTHESIS,
            is_training=False
        )
        g.load_state_dict(ckpt['g_ema'])
        g = copy.deepcopy(g).eval().requires_grad_(False).to(device)

        if args.num_gpus > 1:
            torch.distributed.barrier()
        fid_tracker.calc_fid(g, k_iter=k_iter, save=True)

    if rank == 0:
        fid_tracker.plot_fid()


if __name__ == '__main__':
    import argparse
    import tempfile
    from config import get_cfg_defaults

    parser = argparse.ArgumentParser()
    # parser.add_argument('--truncation', type=float, default=1)
    # parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument("--cfg", type=str, help="path to the configuration file")
    parser.add_argument("--gpus", type=int, default=1, dest='num_gpus')
    parser.add_argument('--out_dir', type=str, default='/tmp/fid_result')
    parser.add_argument('--ckpt', type=str, metavar='CHECKPOINT', help='model ckpt or dir')
    parser.add_argument('--real', action='store_true', default=False, help='get FID between 2 real image dataset')
    parser.add_argument("--debug", action='store_true', default=False, help="whether to use debug mode")

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)

    assert args.num_gpus >= 1
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, cfg=cfg, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, cfg, temp_dir), nprocs=args.num_gpus)
