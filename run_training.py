import os
import sys
import copy
import shutil
import random
import argparse
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
from torch_utils.misc import print_module_summary, constant
from config import get_cfg_defaults, convert_to_dict, override
from dataset import get_dataset, get_dataloader
from models import Generator, Discriminator
from augment import AugmentPipe
from losses import nonsaturating_loss, path_regularize, logistic_loss, d_r1_loss, MaskedRecLoss
from metrics.fid import FIDTracker


class UserError(Exception):
    pass


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


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
        self.num_classes = cfg.num_classes
        self.z_dim = cfg.MODEL.z_dim
        self.batch_gpu = cfg.TRAIN.batch_gpu
        self.device = torch.device(f'cuda:{args.local_rank}')
        self.metrics = cfg.EVAL.metrics.split(',')
        self.fid_tracker = None
        self.sample = None
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

        # Datset
        self.log.info("Prepare dataloader")
        self.loader = get_dataloader(get_dataset(cfg.DATASET, split='train'),
                                     self.batch_gpu, distributed=self.ddp, persistent_workers=True)
        cfg_fakeface = override(cfg.DATASET, dict(dataset='FakeDeepFashionFace', kwargs=None), copy=True)
        self.loader2 = get_dataloader(get_dataset(cfg_fakeface, split='train'),
                                      self.batch_gpu, distributed=self.ddp, persistent_workers=True)

        # Define model
        self.g = Generator(
            self.z_dim,
            cfg.num_classes,
            cfg.resolution,
            extra_channels=cfg.MODEL.extra_channel,
            use_style_encoder=cfg.MODEL.use_style_encoder,
            map_kwargs=cfg.MODEL.G_MAP,
            style_encoder_kwargs=cfg.MODEL.STYLE_ENCODER,
            synthesis_kwargs=cfg.MODEL.G_SYNTHESIS,
            # is_training=True
        ).to(self.device)

        self.d = Discriminator(
            cfg.num_classes,
            cfg.resolution,
            extra_channels=cfg.MODEL.extra_channel
        ).to(self.device)

        self.g_ema = copy.deepcopy(self.g).eval()

        # Define losses
        self.rec_loss = MaskedRecLoss(mask='gaussian', num_channels=1, device=self.device)

        if cfg.DATASET.ADA:
            self.augment_pipe = AugmentPipe(**convert_to_dict(cfg.ADA)).train().requires_grad_(False).to(self.device)
            self.augment_pipe.p.copy_(torch.as_tensor(cfg.DATASET.ADA_p))

        # Define optimizers
        g_reg_ratio = cfg.TRAIN.Greg_every / (cfg.TRAIN.Greg_every + 1)
        d_reg_ratio = cfg.TRAIN.Dreg_every / (cfg.TRAIN.Dreg_every + 1)
        self.g_optim = optim.Adam(self.g.parameters(), lr=cfg.TRAIN.lrate * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
        self.d_optim = optim.Adam(self.d.parameters(), lr=cfg.TRAIN.lrate * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

        # resume from checkpoints if given
        self.start_iter = 0
        if cfg.TRAIN.ckpt:
            print(f'resume model from {cfg.TRAIN.ckpt}')
            ckpt = torch.load(cfg.TRAIN.ckpt, map_location=self.device)

            self.g.load_state_dict(ckpt['g'])
            self.d.load_state_dict(ckpt['d'])
            self.g_ema.load_state_dict(ckpt['g_ema'])

            self.g_optim.load_state_dict(ckpt['g_optim'])
            self.d_optim.load_state_dict(ckpt['d_optim'])

            if 'g_scaler' in ckpt.keys():
                self.g_scaler.load_state_dict(ckpt['g_scaler'])
                self.d_scaler.load_state_dict(ckpt['d_scaler'])

            try:
                self.start_iter = int(Path(cfg.TRAIN.ckpt).stem.split('-')[1])
            except ValueError:
                raise UserError("Fail to parse #iteration from checkpoint filename. Valid format is 'ckpt-<#iter>.pt'")

        # Print network summary tables
        if self.local_rank == 0:
            z = torch.empty([1, self.batch_gpu, self.z_dim], device=self.device).unbind(0)
            c = torch.randint(self.num_classes, (self.batch_gpu,), device=self.device) if self.num_classes > 1 else None
            face = torch.empty([self.batch_gpu, 3, cfg.resolution, cfg.resolution], device=self.device)
            masked_body = torch.empty([self.batch_gpu, 4, cfg.resolution, cfg.resolution], device=self.device)
            img, _ = print_module_summary(self.g, [z, c, face, masked_body])
            print_module_summary(self.d, [img, c])

        if self.ddp:
            self.g = nn.parallel.DistributedDataParallel(
                self.g,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False
            )

            self.d = nn.parallel.DistributedDataParallel(
                self.d,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False
            )

        # init. FID tracker if needed.
        if 'fid' in self.metrics:
            self.fid_tracker = FIDTracker(cfg, self.local_rank, self.num_gpus, self.out_dir)

    def train(self):
        cfg_d = self.cfg.DATASET
        cfg_t = self.cfg.TRAIN
        if self.start_iter >= cfg_t.iteration:
            print("the current iteration already meet your target iteration")
            return

        digits_length = len(str(cfg_t.iteration))
        ema_beta = 0.5 ** (self.batch_gpu * self.num_gpus / (10 * 1000))
        mean_path_length = 0
        stats_keys = ['g', 'd', 'g_rec', 'real_score', 'fake_score', 'mean_path', 'r1', 'path', 'path_length']
        stats = OrderedDict((k, torch.tensor(0.0, dtype=torch.float, device=self.device)) for k in stats_keys)
        fids = None
        ada_p = 0.0
        if cfg_d.ADA and (cfg_d.ADA_target) > 0:
            ada_moments = torch.zeros([2], device=self.device)  # [num_scalars, sum_of_scalars]
            ada_sign = torch.tensor(0.0, dtype=torch.float, device=self.device)

        # To make state_dict consistent in mutli nodes & single node training
        g_module = self.g.module if self.ddp else self.g
        d_module = self.d.module if self.ddp else self.d

        loader = sample_data(self.loader, ddp=self.ddp)
        loader2 = sample_data(self.loader2, ddp=self.ddp)
        if self.local_rank == 0:
            pbar = None

        # main loop
        for i in range(self.start_iter, cfg_t.iteration):
            s = time()
            body_imgs, face_imgs, mask, *args = [x.to(self.device, non_blocking=cfg_d.pin_memory) for x in next(loader)]
            if i % 2 == 0:
                masked_body, face_imgs, mask = [x.to(self.device, non_blocking=cfg_d.pin_memory) for x in next(loader2)]
                masked_body = torch.cat([masked_body, mask], dim=1)
            else:
                masked_body = torch.cat([body_imgs * mask, mask], dim=1)
            fake_label = None

            # D.
            requires_grad(self.g, False)
            requires_grad(self.d, True)

            with autocast(enabled=self.autocast):
                noise = mixing_noise(self.batch_gpu, self.z_dim, cfg_t.style_mixing_prob, self.device)
                fake_img, _ = self.g(noise, labels_in=fake_label, style_in=face_imgs, content_in=masked_body)

                aug_fake_img = self.augment_pipe(fake_img) if cfg_d.ADA else fake_img
                aug_body_imgs = self.augment_pipe(body_imgs) if cfg_d.ADA else body_imgs

                fake_pred = self.d(aug_fake_img, labels_in=fake_label)
                real_pred = self.d(aug_body_imgs, labels_in=fake_label)

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
                aug_body_imgs.requires_grad = True
                with autocast(enabled=self.autocast):
                    real_pred = self.d(aug_body_imgs)
                    r1_loss = d_r1_loss(real_pred, aug_body_imgs)
                    Dreg_loss = cfg_t.r1 / 2 * r1_loss * cfg_t.Dreg_every + 0 * real_pred[0]

                self.d_scaler.scale(Dreg_loss).backward()
                self.d_scaler.step(self.d_optim)
                self.d_scaler.update()
                stats['r1'] = r1_loss.detach()

            requires_grad(self.g, True)
            requires_grad(self.d, False)

            # G.
            with autocast(enabled=self.autocast):
                noise = mixing_noise(self.batch_gpu, self.z_dim, cfg_t.style_mixing_prob, self.device)
                fake_img, _ = self.g(noise, labels_in=fake_label, style_in=face_imgs, content_in=masked_body)

                aug_fake_img = self.augment_pipe(fake_img) if cfg_d.ADA else fake_img
                fake_pred = self.d(aug_fake_img, labels_in=fake_label)
                g_adv_loss = nonsaturating_loss(fake_pred)
                g_rec_loss = self.rec_loss(masked_body[:, :3, :, :], fake_img, mask=mask)  #
                g_loss = g_adv_loss + g_rec_loss
                stats['g'] = g_adv_loss.detach()
                stats['g_rec'] = g_rec_loss.detach()

            self.g.zero_grad(set_to_none=True)
            self.g_scaler.scale(g_loss).backward()
            self.g_scaler.step(self.g_optim)
            self.g_scaler.update()

            if i % cfg_t.Greg_every == 0:
                self.log.debug("Apply regularization to G")
                self.g.zero_grad(set_to_none=True)
                path_batch_size = max(1, self.batch_gpu // cfg_t.path_bs_shrink)

                with autocast(enabled=self.autocast):
                    noise = mixing_noise(path_batch_size, self.z_dim, cfg_t.style_mixing_prob, self.device)

                    fake_img, latents = self.g(
                        noise, labels_in=fake_label, style_in=face_imgs[:path_batch_size], content_in=masked_body[:path_batch_size], return_latents=True)

                    path_loss, mean_path_length, path_lengths = path_regularize(fake_img, latents, mean_path_length)
                    weighted_path_loss = cfg_t.path_reg_gain * cfg_t.Greg_every * path_loss

                    if cfg_t.path_bs_shrink:
                        weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                self.g_scaler.scale(weighted_path_loss).backward()
                self.g_scaler.step(self.g_optim)
                self.g_scaler.update()

                stats['path'] = path_loss.detach()
                stats['path_length'] = path_lengths.mean().detach()
                stats['mean_path'] = mean_path_length.detach()

            accumulate(self.g_ema, g_module, ema_beta)

            # Execute ADA heuristic.
            if cfg_d.ADA and (cfg_d.ADA_target) > 0 and (i % cfg_d.ADA_interval == 0):
                if self.num_gpus > 1:
                    torch.distributed.all_reduce(ada_moments)
                ada_sign = (ada_moments[1] / ada_moments[0]).cpu().numpy()
                adjust = np.sign(ada_sign - cfg_d.ADA_target) * (self.batch_gpu * self.num_gpus * cfg_d.ADA_interval) / (cfg_d.ADA_kimg * 1000)
                self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(constant(0, device=self.device)))
                ada_p = self.augment_pipe.p.item()
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
                reduced_stats = {k: (v / self.num_gpus).item()
                                 for k, v in zip(stats.keys(), losses[0])}
                reduced_stats['ada_p'] = ada_p

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
                        'G-reconstruction': reduced_stats['g_rec'],
                        'Discriminator': reduced_stats['d'],
                        'R1': reduced_stats['r1'],
                        'Path Length Regularization': reduced_stats['path'],
                        'Path Length': reduced_stats['path_length'],
                        'Mean Path Length': reduced_stats['mean_path'],
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
                desc = "d: {d:.4f}; g: {g:.4f}; g_rec: {g_rec:.4f}; r1: {r1:.4f}; path: {path:.4f}; mean path: {mean_path:.4f}; ada_p: {ada_p:.2f}"
                pbar.set_description(desc.format(**reduced_stats))

        if self.local_rank == 0:
            pbar.close()

            if self.fid_tracker:
                self.fid_tracker.plot_fid()

    def _get_sample_data(self):
        cfg_d = self.cfg.DATASET
        sample = misc.EasyDict()
        sample.body_imgs, sample.face_imgs, sample.mask = [], [], []

        sample_setting = dict(xflip=False, pin_memory=False)

        for ds in self.cfg.sample_ds:
            sample_setting['dataset'] = ds
            if ds != self.cfg.DATASET.dataset:
                sample_setting['kwargs'] = None
            sample_cfg = override(self.cfg.DATASET, sample_setting, copy=True)
            ds = get_dataset(sample_cfg, split='val', num_items=(self.n_sample // len(self.cfg.sample_ds)))
            loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
            body_imgs, face_imgs, mask, *args = [x.to(self.device) for x in next(iter(loader))]
            if len(args) == 1:
                # resampling on real dataset
                body_imgs = args[0]

            sample.body_imgs.append(body_imgs)
            sample.face_imgs.append(face_imgs)
            sample.mask.append(mask)

        sample.body_imgs = torch.cat(sample.body_imgs, dim=0)
        sample.face_imgs = torch.cat(sample.face_imgs, dim=0)
        sample.mask = torch.cat(sample.mask, dim=0)
        sample.masked_body = torch.cat([sample.body_imgs * sample.mask, sample.mask], dim=1)
        sample.z = torch.randn(self.n_sample, self.z_dim, device=self.device)
        sample.label = None
        self.log.debug(f"sample vector: {sample.z.shape}")

        return sample

    def sampling(self, idx):
        assert self.local_rank == 0
        if self.sample is None:
            self.sample = self._get_sample_data()

        cfg = self.cfg.DATASET
        with torch.no_grad():
            self.g_ema.eval()
            samples, _ = self.g_ema([self.sample.z], labels_in=self.sample.label, noise_mode='const',
                                    style_in=self.sample.face_imgs, content_in=self.sample.masked_body)
            b, c, h, w = samples.shape

            if cfg.dataset == 'MultiChannelDataset':
                assert c == sum(cfg.channels)
                samples = torch.split(samples, cfg.channels, dim=1)
                samples = [(x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x)for x in samples]
                samples = torch.cat(samples, dim=2)
            else:
                samples = torch.stack([self.sample.face_imgs, self.sample.body_imgs, samples], dim=0)
                samples = torch.transpose(samples, 0, 1).reshape(3 * b, c, h, w)

            utils.save_image(
                samples,
                self.out_dir / f'samples/fake-{idx}.png',
                nrow=int(self.n_sample ** 0.5) * 3,
                normalize=True,
                value_range=(-1, 1),
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='torch.distributed.launch')
    parser.add_argument('-c', '--cfg', help="path to the configuration file", metavar='PATH')
    parser.add_argument('-o', '--out_dir', metavar='PATH',
                        help="path to output directory. If not set, auto. set to subdirectory of outdir in configuration")
    parser.add_argument('--local_rank', type=int, default=0, metavar='INT', help="Automatically given by %(prog)s")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--nobench', default=True, action='store_false', dest='cudnn_benchmark', help="disable cuDNN benchmarking")
    parser.add_argument('--autocast', default=False, action='store_true', help="whether to use `torch.cuda.amp.autocast")
    parser.add_argument('--gradscale', default=False, action='store_true', help="whether to use gradient scaler")
    parser.add_argument('--no-wandb', default=True, action='store_false', dest='wandb', help="disable wandb logging")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.cfg:
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
            print(f"initialize wandb project: {Path(args.cfg).stem}")

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
                    raise UserError(f"non-increased step in log cal is not allowed in Wandb."
                                    f"Please unset WANDB_RESUME env variable or set correct checkpoint.")

            args.wandb_id = run.id

        misc.prepare_training(args, cfg)
        shutil.copy(args.cfg, args.out_dir)
        print(cfg)

    logger = misc.create_logger(**vars(args))

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
