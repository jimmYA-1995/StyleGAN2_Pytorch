import os
import sys
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
from config import get_cfg_defaults, override
from dataset import get_dataset, ConditionalBatchSampler
from .calc_inception import load_patched_inception_v3


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
        self.log = logging.getLogger(f'GPU{rank}')
        self.pbar = pbar(rank)
        self.out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir

        # metrics
        self.real_means = None  # [ndarray(2048) float32] * num_class
        self.real_covs = None  # [ndarray(2048, 2048) float64] * num_class
        self.classes = []
        self.k_iters = []
        self.fids = []

        start = time.time()
        self.log.info("load inceptionV3 model...")
        if num_gpus == 1:
            self.inceptionV3 = load_patched_inception_v3().eval().to(self.device)
        else:
            if self.rank != 0:
                torch.distributed.barrier()
            self.inceptionV3 = load_patched_inception_v3().eval().to(self.device)
            if self.rank == 0:
                torch.distributed.barrier()
        self.log.info("load inceptionV3 model complete ({:.2f} sec)".format(time.time() - start))

        # get features for real images
        if self.cfg.inception_cache:
            self.log.info("load inception from cache file")
            with open(self.cfg.inception_cache, 'rb') as f:
                embeds = pickle.load(f)
                self.real_means = embeds['mean']
                self.real_covs = embeds['cov']
                self.classes = embeds['classes']
        else:
            self.real_means, self.real_covs = self.extract_feature_from_images(cfg)
            if self.rank == 0:
                self.log.info(f"save inception cache to {self.out_dir / 'inception_cache.pkl'}")
                with open(self.out_dir / 'inception_cache.pkl', 'wb') as f:
                    pickle.dump(dict(mean=self.real_means, cov=self.real_covs, classes=self.classes), f)

        cfg_val = cfg.DATASET.clone()
        if self.cfg.dataset:
            val_setting = dict(dataset=self.cfg.dataset, xflip=False, pin_memory=False)
            if self.cfg.dataset != cfg_val.dataset:
                val_setting['kwargs'] = None
            override(cfg_val, val_setting)
        self.val_dataset = get_dataset(cfg_val, split='val')
        self.log.info(f"validation dataset: {self.val_dataset.classes}")

    @classmethod
    def calc_fid(cls, real_mean, real_cov, sample_mean, sample_cov, eps=1e-6):
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

    def __call__(self, candidate, k_iter=0, save=False, eps=1e-6):
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
            fid = FIDTracker.calc_fid(self.real_means[i], self.real_covs[i], sample_mean, sample_cov, eps=eps)
            fids.append(fid)

        total_time = time.time() - start
        self.log.info(f'FID on {1000 * k_iter: .0f} iterations: "{fids}". [costs {total_time: .2f} sec(s)]')
        self.k_iters.append(k_iter)
        self.fids.append(fids)

        if self.rank == 0 and save:
            with open(self.out_dir / 'fid.txt', 'a+') as f:
                f.write(f'{k_iter}: {fids}\n')

            # compatible with NVLab
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
        image_bs = 8
        means, covs = [], []
        dataset = get_dataset(cfg.DATASET, split='all')
        dataset.xflip = False

        if not self.classes:
            self.classes = dataset.classes
        elif self.classes != dataset.classes:
            self.log.error(f"classes are not matched. {self.classes} v.s {dataset.classes}")
            sys.exit(1)

        for i, c in enumerate(dataset.classes):
            start = time.time()
            data_size = len(dataset) if len(dataset.classes) == 1 else dataset.size[c]
            num_items = min(data_size, self.cfg.n_sample)
            self.log.info(f'extract features from {num_items} real "{c}" images...')

            batch_sampler = ConditionalBatchSampler(dataset,
                                                    class_indices=[i],
                                                    sample_per_class=image_bs,
                                                    no_repeat=True,
                                                    num_gpus=self.num_gpus,
                                                    rank=self.rank)

            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_sampler=batch_sampler,
                                                     num_workers=dataset.cfg.workers)

            features = []
            loader = iter(dataloader)

            # TODO: update progress bar to samples instead of batch
            for _ in self.pbar(len(batch_sampler)):
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
        ds = self.val_dataset
        classes = ds.classes
        assert len(self.classes) == len(classes)
        self.log.info("get FID between {}".format(
            ', '.join(f"{x} v.s {y}" for x, y in zip(self.classes, classes))))

        sample_means, sample_covs = [], []
        items_per_gpu = self.cfg.n_sample // self.num_gpus + int(self.cfg.n_sample % self.num_gpus != 0)

        for i, c in enumerate(classes):
            data_size = len(ds) if len(classes) == 1 else ds.size[c]
            if data_size < self.cfg.n_sample:
                self.log.warn(f"#conditional inputs for {c} < required #samples.")

            batch_sampler = ConditionalBatchSampler(ds,
                                                    class_indices=[i],
                                                    sample_per_class=self.cfg.batch_size,
                                                    num_items=items_per_gpu,
                                                    shuffle=True,
                                                    num_gpus=self.num_gpus,
                                                    rank=self.rank)

            dataloader = torch.utils.data.DataLoader(ds,
                                                     batch_sampler=batch_sampler,
                                                     num_workers=ds.cfg.workers,
                                                     worker_init_fn=ds.__class__.worker_init_fn)

            features = []
            loader = iter(dataloader)
            for i in self.pbar(len(batch_sampler)):
                body_imgs, face_imgs, mask, heatmaps = [x.to(self.device) for x in next(loader)]
                # if len(args) == 2:
                #     fake_body, mask = args
                #     masked_body = torch.cat([(fake_body * mask), mask], dim=1)
                # else:
                masked_body = torch.cat([(body_imgs * mask), mask], dim=1)
                latent = torch.randn(body_imgs.shape[0], generator.z_dim, device=self.device)

                imgs, _ = generator([latent], style_in=face_imgs, content_in=masked_body, pose_in=heatmaps, noise_mode='const')

                feature = self.inceptionV3(imgs[:, :3, :, :])[0].view(imgs.shape[0], -1)
                if self.num_gpus > 1:
                    _features = []
                    for src in range(self.num_gpus):
                        y = feature.clone()
                        torch.distributed.broadcast(y, src=src)
                        _features.append(y)
                    feature = torch.stack(_features, dim=1).flatten(0, 1)
                features.append(feature)

            features = torch.cat(features, 0)[:self.cfg.n_sample].cpu().numpy()
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
        plt.legend(self.classes, loc='upper right')
        plt.savefig(self.out_dir / 'fid.png')


def subprocess_fn(rank, args, cfg, temp_dir):
    from models import Generator
    if args.num_gpus > 1:
        torch.cuda.set_device(rank)
        init_method = f"file://{os.path.join(temp_dir, '.torch_distributed_init')}"
        torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)
    args.local_rank = rank  # compatibility
    misc.create_logger(**vars(args))
    device = torch.device('cuda', rank) if args.num_gpus > 1 else 'cuda'
    fid_tracker = FIDTracker(cfg, rank, args.num_gpus, args.out_dir)

    if args.real:
        fid_tracker(cfg, save=args.save)
        return

    if args.ckpt is None:
        if rank == 0:
            print("only get statistics on real data. return...")
        return

    args.ckpt = Path(args.ckpt)
    args.ckpt = sorted(p for p in args.ckpt.glob('*.pt')) if args.ckpt.is_dir() else [args.ckpt]
    for ckpt in args.ckpt:
        print(f"load {str(ckpt)}")
        k_iter = int(str(ckpt.name)[5:11]) / 1e3
        ckpt = torch.load(ckpt)

        g = Generator(
            cfg.MODEL.z_dim,
            cfg.num_classes,
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
        fid_tracker(g, k_iter=k_iter, save=args.save)

    if rank == 0:
        fid_tracker.plot_fid()


if __name__ == '__main__':
    import argparse
    import tempfile

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
            os.environ['OMP_NUM_THREADS'] = '1'  # for scipy performance issue
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, cfg, temp_dir), nprocs=args.num_gpus)
