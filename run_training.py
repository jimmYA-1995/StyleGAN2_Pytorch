import os
import sys
import copy
import shutil
import random
import argparse
import warnings
from time import time
from pathlib import Path
from collections import OrderedDict

import wandb
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torchvision import utils
from torch.cuda.amp import autocast, GradScaler

import misc
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from torch_utils.misc import constant
from config import get_cfg_defaults, convert_to_dict, override
from dataset import get_dataset, get_dataloader
from models.utils import create_model, ema, print_module_summary
from augment import AugmentPipe
from losses import nonsaturating_loss, path_regularize, logistic_loss, d_r1_loss
from metrics.fid import FIDTracker


class UserError(Exception):
    pass


def launch_wandb(cfg):
    run = wandb.init(
        project=f'stylegan2-{Path(args.cfg).stem}',
        config=convert_to_dict(cfg),
        notes=cfg.description,
        tags=['finetune'] if cfg.TRAIN.ckpt else None,
    )

    if cfg.name:
        run.name = cfg.name

    if run.resumed:
        assert cfg.TRAIN.ckpt
        try:
            start_iter = int(Path(cfg.TRAIN.ckpt).stem.split('-')[1])
        except ValueError:
            raise UserError("Fail to parse #iteration from checkpoint filename. Valid format is 'ckpt-<#iter>.pt'")

        if run.starting_step != start_iter:
            warnings.warn("non-increased step in log cal is not allowed in Wandb."
                          "It will cause wandb skip logging until last step in previous run")

    return run


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader, ddp=False):
    while True:
        epoch = 0
        if ddp:
            assert isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler)
            loader.sampler.set_epoch(epoch)

        for batch in loader:
            yield batch

        epoch += 1


def mixing_noise(batch, latent_dim, prob, device):
    n_noise = 2 if random.random() < prob else 1
    return torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)


class Trainer():
    def __init__(self, args, cfg, logger):
        self.cfg = cfg
        self.log = logger
        self.num_gpus = args.num_gpus
        self.local_rank = args.local_rank
        self.ddp = args.num_gpus > 1
        self.out_dir = args.out_dir
        self.use_wandb = args.wandb
        self.n_sample = cfg.n_sample
        self.classes = cfg.classes
        self.z_dim = cfg.MODEL.z_dim
        self.batch_gpu = cfg.TRAIN.batch_gpu
        self.device = torch.device(f'cuda:{args.local_rank}')
        self.metrics = cfg.EVAL.metrics.split(',')
        self.fid_tracker = None
        self.sample = None
        self.start_iter = 0
        self.autocast = args.autocast

        # reproducibility
        random.seed(args.seed * args.num_gpus + args.local_rank)
        np.random.seed(args.seed * args.num_gpus + args.local_rank)
        torch.manual_seed(args.seed * args.num_gpus + args.local_rank)

        # performance setting
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        # torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
        # torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
        if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9']):
            conv2d_gradfix.enabled = True
        else:
            self.log.warn("torch version not later than 1.7. disable conv2d_gradfix.")
            conv2d_gradfix.enabled = False
        grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

        self.g_scaler = GradScaler(enabled=args.gradscale)
        self.d_scaler = GradScaler(enabled=args.gradscale)

        # dataset
        self.log.info("Prepare dataloader")
        self.loader = get_dataloader(get_dataset(cfg.DATASET, split='all'),
                                     self.batch_gpu, distributed=self.ddp, persistent_workers=True)

        # Define model
        self.g, self.d = create_model(cfg)
        self.g_ema = copy.deepcopy(self.g).eval()

        if cfg.DATASET.ADA:
            self.augment_pipe = AugmentPipe(**cfg.ADA).train().requires_grad_(False).to(self.device)
            self.augment_pipe.p.copy_(torch.as_tensor(cfg.DATASET.ADA_p))

        # Define optimizers (Lazy regularizer)
        g_reg_ratio = cfg.TRAIN.Greg_every / (cfg.TRAIN.Greg_every + 1)
        d_reg_ratio = cfg.TRAIN.Dreg_every / (cfg.TRAIN.Dreg_every + 1)
        self.g_optim = optim.Adam(self.g.parameters(), lr=cfg.TRAIN.lrate * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
        self.d_optim = optim.Adam(self.d.parameters(), lr=cfg.TRAIN.lrate * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

        self.resume_from_checkpoint(cfg.TRAIN.ckpt)

        # Print network summary tables
        if self.local_rank == 0:
            z = torch.empty([self.batch_gpu, self.z_dim], device=self.device)
            c = None  # torch.randint(self.num_classes, (self.batch_gpu,), device=self.device) if self.num_classes > 1 else None
            heatmaps = torch.empty([self.batch_gpu, 17, 256, 256], device=self.device)
            imgs = print_module_summary(self.g, [z, c, heatmaps])
            print_module_summary(self.d, [imgs, c])

        if self.ddp:
            self.g = nn.parallel.DistributedDataParallel(
                self.g,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                # find_unused_parameters=True
            )

            self.d = nn.parallel.DistributedDataParallel(
                self.d,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False
            )

        # Metrics
        if 'fid' in self.metrics:
            self.fid_tracker = FIDTracker(cfg, self.local_rank, self.num_gpus, self.out_dir)

    def resume_from_checkpoint(self, ckpt_path):
        if not ckpt_path:
            return

        print(f'resume model from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.g.load_state_dict(ckpt['g'])
        self.d.load_state_dict(ckpt['d'])
        self.g_ema.load_state_dict(ckpt['g_ema'])

        self.g_optim.load_state_dict(ckpt['g_optim'])
        self.d_optim.load_state_dict(ckpt['d_optim'])

        if 'g_scaler' in ckpt.keys():
            self.g_scaler.load_state_dict(ckpt['g_scaler'])
            self.d_scaler.load_state_dict(ckpt['d_scaler'])

        self.mean_path_length = ckpt.get('mean_path_lenght', 0.0)
        self.ada_p = ckpt.get('ada_p', 0.0)
        try:
            self.start_iter = int(Path(ckpt_path).stem.split('-')[1])
        except ValueError:
            raise UserError("Fail to parse #iteration from checkpoint filename. Valid format is 'ckpt-<#iter>.pt'")

    def train(self):
        cfg_d = self.cfg.DATASET
        cfg_t = self.cfg.TRAIN
        if self.start_iter >= cfg_t.iteration:
            print("the current iteration already meet your target iteration")
            return

        digits_length = len(str(cfg_t.iteration))
        ema_beta = 0.5 ** (self.batch_gpu * self.num_gpus / (10 * 1000))
        stats_keys = ['g', 'd', 'real_score', 'fake_score', 'mean_path', 'r1', 'path', 'path_length']
        stats = OrderedDict((k, torch.tensor(0.0, dtype=torch.float, device=self.device)) for k in stats_keys)
        fids = None
        if cfg_d.ADA and (cfg_d.ADA_target) > 0:
            ada_moments = torch.zeros([2], device=self.device)  # [num_scalars, sum_of_scalars]
            ada_sign = torch.tensor(0.0, dtype=torch.float, device=self.device)

        # To make state_dict consistent in mutli nodes & single node training
        g_module = self.g.module if self.ddp else self.g
        d_module = self.d.module if self.ddp else self.d

        loader = sample_data(self.loader, ddp=self.ddp)
        if self.local_rank == 0:
            pbar = None

        # main loop
        for i in range(self.start_iter, cfg_t.iteration):
            s = time()
            real_face, real_human, heatmaps = [x.to(self.device, non_blocking=cfg_d.pin_memory) for x in next(loader)]
            real_imgs = torch.cat([real_face, real_human], dim=1)
            fake_label = None

            # D.
            requires_grad(self.g, False)
            requires_grad(self.d, True)

            with autocast(enabled=self.autocast):
                z = torch.randn(self.batch_gpu, self.z_dim, device=self.device)
                fake_imgs = self.g(z, c=fake_label, pose=heatmaps)

                aug_fake_imgs = self.augment_pipe(fake_imgs) if cfg_d.ADA else fake_imgs
                aug_real_imgs = self.augment_pipe(real_imgs) if cfg_d.ADA else real_imgs

                fake_pred = self.d(aug_fake_imgs, labels_in=fake_label)
                real_pred = self.d(aug_real_imgs, labels_in=fake_label)

                if cfg_d.ADA and (cfg_d.ADA_target) > 0:
                    ada_moments[0].add_(torch.ones_like(real_pred).sum())
                    ada_moments[1].add_(real_pred.sign().detach().flatten().sum())

                d_loss = logistic_loss(real_pred, fake_pred)

            stats['d'] = d_loss.detach()
            stats['real_score'] = real_pred.mean().detach()
            stats['fake_score'] = fake_pred.mean().detach()

            self.d.zero_grad(set_to_none=True)
            self.d_scaler.scale(d_loss).backward()
            self.d_scaler.step(self.d_optim)
            self.d_scaler.update()

            if i % cfg_t.Dreg_every == 0:
                self.d.zero_grad(set_to_none=True)
                aug_real_imgs.requires_grad = True
                with autocast(enabled=self.autocast):
                    real_pred = self.d(aug_real_imgs)
                    r1_loss = d_r1_loss(real_pred, aug_real_imgs)
                    Dreg_loss = cfg_t.r1 / 2 * r1_loss * cfg_t.Dreg_every + 0 * real_pred[0]

                self.d_scaler.scale(Dreg_loss).backward()
                self.d_scaler.step(self.d_optim)
                self.d_scaler.update()
                stats['r1'] = r1_loss.detach()

            requires_grad(self.g, True)
            requires_grad(self.d, False)

            # G.
            with autocast(enabled=self.autocast):
                z = torch.randn(self.batch_gpu, self.z_dim, device=self.device)
                fake_imgs = self.g(z, c=fake_label, pose=heatmaps)

                aug_fake_imgs = self.augment_pipe(fake_imgs) if cfg_d.ADA else fake_imgs
                fake_pred = self.d(aug_fake_imgs, labels_in=fake_label)
                g_adv_loss = nonsaturating_loss(fake_pred)
                g_loss = g_adv_loss
                stats['g'] = g_adv_loss.detach()

            self.g.zero_grad(set_to_none=True)
            self.g_scaler.scale(g_loss).backward()
            self.g_scaler.step(self.g_optim)
            self.g_scaler.update()

            if i % cfg_t.Greg_every == 0:
                self.log.debug("Apply regularization to G")
                self.g.zero_grad(set_to_none=True)
                path_batch_size = max(1, self.batch_gpu // cfg_t.path_bs_shrink)

                with autocast(enabled=self.autocast):
                    z = mixing_noise(path_batch_size, self.z_dim, cfg_t.style_mixing_prob, self.device)[0]
                    fake_imgs, ws = self.g(z, c=fake_label, pose=heatmaps[:path_batch_size], return_dlatent=True)

                    fake_face, _ = torch.split(fake_imgs, [3, 3], dim=1)
                    # PPL regularization only on face
                    path_loss, self.mean_path_length, path_lengths = path_regularize(fake_face, ws[0], self.mean_path_length)
                    weighted_path_loss = cfg_t.path_reg_gain * cfg_t.Greg_every * path_loss

                    if cfg_t.path_bs_shrink:
                        weighted_path_loss += 0 * fake_face[0, 0, 0, 0]

                self.g_scaler.scale(weighted_path_loss).backward()
                self.g_scaler.step(self.g_optim)
                self.g_scaler.update()

                stats['path'] = path_loss.detach()
                stats['path_length'] = path_lengths.mean().detach()
                stats['mean_path'] = self.mean_path_length.detach()

            ema(self.g_ema, g_module, ema_beta)

            # Execute ADA heuristic.
            if cfg_d.ADA and (cfg_d.ADA_target) > 0 and (i % cfg_d.ADA_interval == 0):
                if self.num_gpus > 1:
                    torch.distributed.all_reduce(ada_moments)
                ada_sign = (ada_moments[1] / ada_moments[0]).cpu().numpy()
                adjust = np.sign(ada_sign - cfg_d.ADA_target) * (self.batch_gpu * self.num_gpus * cfg_d.ADA_interval) / (cfg_d.ADA_kimg * 1000)
                self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(constant(0, device=self.device)))
                self.ada_p = self.augment_pipe.p.item()
                ada_moments = torch.zeros_like(ada_moments)

            if self.fid_tracker is not None and (i == 0 or (i + 1) % self.cfg.EVAL.FID.every == 0):
                k_iter = (i + 1) / 1000
                self.g_ema.eval()
                fids = self.fid_tracker(self.g_ema, k_iter, save=True)

            # reduce loss
            with torch.no_grad():
                losses = [torch.stack(list(stats.values()), dim=0)]
                if self.num_gpus > 1:
                    torch.distributed.reduce_multigpu(losses, dst=0)

            if self.local_rank == 0:
                reduced_stats = {k: (v / self.num_gpus).item() for k, v in zip(stats.keys(), losses[0])}
                reduced_stats['ada_p'] = self.ada_p

                if i == 0 or (i + 1) % cfg_t.sample_every == 0:
                    sample_iter = 'init' if i == 0 else str(i + 1).zfill(digits_length)
                    self.sampling(sample_iter)

                if (i + 1) % cfg_t.save_ckpt_every == 0:
                    ckpt_iter = str(i + 1).zfill(digits_length)
                    ckpt_dir = self.out_dir / 'checkpoints'
                    snapshot = {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': self.g_ema.state_dict(),
                        'g_optim': self.g_optim.state_dict(),
                        'd_optim': self.d_optim.state_dict(),
                        'g_scaler': self.g_scaler.state_dict(),
                        'd_scaler': self.d_scaler.state_dict(),
                    }
                    torch.save(snapshot, ckpt_dir / f'ckpt-{ckpt_iter}.pt')
                    ckpt_paths = list(ckpt_dir.glob('*.pt'))
                    if cfg_t.ckpt_max_keep != -1 and len(ckpt_paths) > cfg_t.ckpt_max_keep:
                        ckpt_idx = sorted([int(str(p.name)[5:5 + digits_length]) for p in ckpt_paths])
                        os.remove(ckpt_dir / f'ckpt-{str(ckpt_idx[0]).zfill(digits_length)}.pt')

                # update dashboard info. and progress bar
                if self.use_wandb:
                    wandb_stats = {
                        'training time': time() - s,
                        'Generator': reduced_stats['g'],
                        'Discriminator': reduced_stats['d'],
                        'R1': reduced_stats['r1'],
                        'Path Length Regularization(face)': reduced_stats['path'],
                        'Path Length(face)': reduced_stats['path_length'],
                        'Mean Path Length(face)': reduced_stats['mean_path'],
                        'Real Score': reduced_stats['real_score'],
                        'Fake Score': reduced_stats['fake_score'],
                        'ADA probability': reduced_stats['ada_p'],
                    }

                    if self.fid_tracker is not None and fids is not None:
                        wandb_stats['FID'] = fids[0]  # one class for now

                    if cfg_d.ADA and cfg_d.ADA_target > 0:
                        wandb_stats['Real Sign'] = ada_sign.item()

                    wandb.log(data=wandb_stats, step=i)

                if pbar is None:
                    print(f"1st iter: {time() - s} sec")
                    pbar = tqdm(total=cfg_t.iteration, initial=i, dynamic_ncols=True, smoothing=0, colour='yellow')

                pbar.update(1)
                desc = "d: {d:.4f}; g: {g:.4f}; r1: {r1:.4f}; path: {path:.4f}; mean path: {mean_path:.4f}; ada_p: {ada_p:.2f}"
                pbar.set_description(desc.format(**reduced_stats))

        if self.local_rank == 0:
            pbar.close()

            if self.fid_tracker:
                self.fid_tracker.plot_fid()

    def _get_sample_data(self):
        sample_cfg = self.cfg.DATASET.clone()
        override(sample_cfg.kwargs, dict(sample=True))
        sample_ds = get_dataset(sample_cfg, split='all')
        sample = misc.EasyDict()

        loader = torch.utils.data.DataLoader(sample_ds, batch_size=self.n_sample, shuffle=False)
        sample.vis_kp, sample.heatmaps = [x.to(self.device) for x in next(iter(loader))]
        sample.z = torch.randn(self.n_sample, self.z_dim, device=self.device)
        sample.label = None
        self.log.debug(f"sample vector: {sample.z.shape}")

        return sample

    def sampling(self, idx):
        assert self.local_rank == 0
        if self.sample is None:
            self.sample = self._get_sample_data()

        with torch.no_grad():
            self.g_ema.eval()
            samples = self.g_ema(self.sample.z, c=self.sample.label, pose=self.sample.heatmaps, noise_mode='const')

            s = torch.split(samples, [3, 3], dim=1) + (self.sample.vis_kp,)
            samples = torch.stack(s, dim=0)  # add visualize pose
            samples = torch.transpose(samples, 0, 1).reshape(len(s) * self.n_sample, *s[0].shape[1:])

            utils.save_image(
                samples,
                self.out_dir / f'samples/fake-{idx}.png',
                nrow=int(self.n_sample ** 0.5) * len(s),
                normalize=True,
                value_range=(-1, 1),
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='torch.distributed.launch')
    parser.add_argument('--cfg', metavar='FILE', help="path to the config file")
    parser.add_argument(
        '--out_dir',
        metavar='PATH',
        help="Path to output directory. If not given, it will automatically "
        "assign a subdirectory under output directory defined by config"
    )
    parser.add_argument('--local_rank', type=int, default=0, metavar='INT', help="Automatically given by %(prog)s")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--nobench', default=True, action='store_false', dest='cudnn_benchmark', help="disable cuDNN benchmarking")
    parser.add_argument('--autocast', default=False, action='store_true', help="whether to use `torch.cuda.amp.autocast")
    parser.add_argument('--gradscale', default=False, action='store_true', help="whether to use gradient scaler")
    parser.add_argument('--no-wandb', default=True, action='store_false', dest='wandb', help="disable wandb logging")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    args.num_gpus = torch.cuda.device_count()
    if args.num_gpus > 1:
        assert 0 <= args.local_rank < args.num_gpus, 'Recommend one process per device'
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', rank=args.local_rank, world_size=args.num_gpus)

        assert torch.distributed.is_available() and torch.distributed.is_initialized()
        torch.distributed.barrier()

    if args.num_gpus == 1 or args.local_rank == 0:
        args.wandb_id = 'noWandB'
        if args.wandb:
            run = launch_wandb(cfg)
            args.wandb_id = run.id

        misc.setup_outdir(args, cfg)
        shutil.copy(args.cfg, args.out_dir)
        print(cfg)

    logger = misc.setup_logger(**vars(args))

    t = time()
    logger.info("initialize trainer...")
    trainer = Trainer(args, cfg, logger)
    logger.info(f"trainer initialized. ({time() - t :.2f} sec)")

    cfg.freeze()
    trainer.train()

    if args.local_rank == 0:
        if args.wandb:
            run.finish()

        (args.out_dir / 'finish.txt').touch()
