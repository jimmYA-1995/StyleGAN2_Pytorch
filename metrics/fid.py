import os
import time
import json
import pickle
import logging
from pathlib import Path

import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm import trange

import misc
from dataset import get_dataset, ResamplingDatasetV2
from .calc_inception import load_patched_inception_v3


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def pbar(rank, force=False):
    def range_wrapper(*args, **kwargs):
        return trange(*args, **kwargs) if rank == 0 or force else range(*args, **kwargs)
    return range_wrapper


class FIDTracker():
    def __init__(self, cfg, rank, num_gpus, out_dir):
        print(f"[{os.getpid()}] rank: {rank}({num_gpus} gpus)")

        self.rank = rank
        self.num_gpus = num_gpus
        self.device = torch.device('cuda', rank) if num_gpus > 1 else 'cuda'
        self.cfg = cfg.EVAL.FID
        self.cfg_d = cfg.DATASET
        self.log = logging.getLogger(f'GPU{rank}')
        self.out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir

        # metrics
        self.real_means = None  # [ndarray(2048) float32] * num_class
        self.real_covs = None  # [ndarray(2048, 2048) float64] * num_class
        self.idx_to_class = []
        self.k_iters = []
        self.fids = []

        self.resolution = cfg.resolution
        self.val_dataset = None  # dataset for model
        self.latent_size = cfg.MODEL.z_dim
        self.conditional = True if cfg.num_classes > 1 else False
        self.num_classes = cfg.num_classes
        self.model_bs = 8
        self.pbar = pbar(rank)

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
        if cfg.EVAL.FID.inception_cache:
            self.log.info("load inception from cache file")
            with open(cfg.EVAL.FID.inception_cache, 'rb') as f:
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

    @classmethod
    def _calc_fid(cls, real_mean, real_cov, sample_mean, sample_cov, eps=1e-6):
        cov_sqrt, _ = scipy.linalg.sqrtm(sample_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = scipy.linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            print("cov_sqrt is complex number")
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)
        return mean_norm + trace

    def calc_fid(self, candidate, k_iter=0, save=False, eps=1e-6):
        assert self.real_means is not None and self.real_covs is not None
        start = time.time()
        fids = []

        # single dispatch
        if isinstance(candidate, torch.nn.Module):
            self.log.info(f"get FID on {k_iter * 1000} iteration") 
            sample_means, sample_covs = self.extract_feature_from_model(candidate)
        else:
            self.log.info(f"get FID on images from {candidate.DATASET.roots[0]}")
            sample_means, sample_covs = self.extract_feature_from_images(candidate)
        assert len(sample_means) == len(self.real_means) == len(sample_covs) == len(self.real_covs)

        for i, (sample_mean, sample_cov) in enumerate(zip(sample_means, sample_covs)):
            fid = FIDTracker._calc_fid(self.real_means[i], self.real_covs[i], sample_mean, sample_cov, eps=eps)
            fids.append(fid)

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
        image_bs = 8
        for i in range(self.num_classes):
            start = time.time()
            self.idx_to_class.append(None)  # no class name now
            # self.log.info(f'extract features from real "{self.idx_to_class[i]}" images...')
            dataset = get_dataset(cfg.DATASET, cfg.resolution, split='train')
            num_items = min(len(dataset), self.cfg.n_sample)
            self.log.info(f"total: {num_items} real images")

            item_subset = [(i * self.num_gpus + self.rank) % num_items
                           for i in range((num_items - 1) // self.num_gpus + 1)]
            self.log.debug(f"#item subset: {len(item_subset)}")

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=item_subset, batch_size=image_bs, num_workers=2)

            features = []
            loader = iter(dataloader)
            n_batch = len(item_subset) // image_bs + int(len(item_subset) % image_bs != 0)

            for _ in self.pbar(n_batch):
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
                features.append(feature)

            features = torch.cat(features, 0)[:num_items].cpu().numpy()
            means.append(np.mean(features, 0))
            covs.append(np.cov(features, rowvar=False))

            self.log.info(f"complete({time.time() - start :.2f} secs). "
                          f"total extracted features: {features.shape[0]}")
        return means, covs

    @torch.no_grad()
    def extract_feature_from_model(self, generator):
        if self.val_dataset is None:
            # self.val_dataset = ResamplingDataset(self.cfg_d, self.resolution)
            self.val_dataset = ResamplingDatasetV2(self.cfg_d, self.resolution, split='val')
            # self.val_dataset = get_dataset(self.cfg_d, self.resolution, split='val')
            self.log.info(f"validation data samples: {len(self.val_dataset)}")
            assert self.cfg.n_sample <= len(self.val_dataset)

        sample_means, sample_covs = [], []
        num_sample = self.cfg.n_sample // self.num_gpus
        n_batch = num_sample // self.model_bs
        resid = num_sample % self.model_bs
        num_items = min(len(self.val_dataset), self.cfg.n_sample)
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

            for i in self.pbar(n_batch + 1):
                batch = resid if i == n_batch else self.model_bs
                if batch == 0:
                    continue

                body_imgs, face_imgs, mask = [x[:batch].to(self.device) for x in next(loader)]
                masked_body = torch.cat(((body_imgs * mask), mask), dim=1)
                latent = torch.randn(batch, self.latent_size, device=self.device)
                fake_label = torch.LongTensor([class_idx] * batch).to(self.device)

                imgs, _ = generator([latent], labels_in=fake_label, style_in=face_imgs, content_in=masked_body, noise_mode='const')

                feature = self.inceptionV3(imgs[:, :3, :, :])[0].view(imgs.shape[0], -1)
                if self.num_gpus > 1:
                    _features = []
                    for src in range(self.num_gpus):
                        y = feature.clone()
                        torch.distributed.broadcast(y, src=src)
                        _features.append(y)
                    feature = torch.stack(_features, dim=1).flatten(0, 1)
                features.append(feature)

            features = torch.cat(features, 0)[:num_items].cpu().numpy()
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
    if args.num_gpus > 1:
        os.environ['OMP_NUM_THREADS'] = '1'  # for scipy performance issue
        torch.cuda.set_device(rank)
        init_method = f"file://{os.path.join(temp_dir, '.torch_distributed_init')}"
        torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)
    args.local_rank = rank  # compatibility
    misc.create_logger(**vars(args))
    device = torch.device('cuda', rank) if args.num_gpus > 1 else 'cuda'
    fid_tracker = FIDTracker(cfg, rank, args.num_gpus, args.out_dir)

    if args.real:
        fid_tracker.calc_fid(cfg, save=args.save)
        return

    if args.ckpt is None:
        if rank == 0:
            print("only get statistics on real data. return...")
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
            map_kwargs=cfg.MODEL.G_MAP,
            style_encoder_kwargs=cfg.MODEL.STYLE_ENCODER,
            synthesis_kwargs=cfg.MODEL.G_SYNTHESIS,
            is_training=False
        )
        g.load_state_dict(ckpt['g_ema'])
        g = g.eval().requires_grad_(False).to(device)

        if args.num_gpus > 1:
            torch.distributed.barrier()
        fid_tracker.calc_fid(g, k_iter=k_iter, save=args.save)

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
    parser.add_argument('--ckpt', type=str, default=None, metavar='CHECKPOINT', help='model ckpt or dir')
    parser.add_argument('--real', action='store_true', default=False, help='get FID between 2 real image dataset')
    parser.add_argument('--save', action='store_true', default=False, help='save fid.txt')
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
