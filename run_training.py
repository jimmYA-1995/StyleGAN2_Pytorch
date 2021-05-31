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

import misc
from config import get_cfg_defaults, convert_to_dict
from dataset import get_dataset, get_dataloader, ResamplingDataset
from models import Generator, Discriminator
from losses import nonsaturating_loss, path_regularize, logistic_loss, d_r1_loss, MaskedRecLoss
from metrics.fid import FIDTracker


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


class Trainer():
    def __init__(self, args, cfg, logger):
        self.cfg = cfg
        self.log = logger
        self.num_gpus = args.num_gpus
        self.local_rank = args.local_rank
        self.ddp = args.num_gpus > 1
        self.out_dir = args.out_dir
        self.use_wandb = args.wandb
        self.n_sample = cfg.N_SAMPLE
        self.num_classes = cfg.N_CLASSES
        self.latent = cfg.MODEL.LATENT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        self.device = torch.device(f'cuda:{args.local_rank}')
        self.metrics = cfg.EVAL.METRICS.split(',')
        self.fid_tracker = None
        self.sample = None

        # Datset
        t = time()
        print("get dataloader ...", end='\r')
        self.loader = get_dataloader(cfg, self.batch_size, distributed=self.ddp)
        print(f"get dataloader complete ({time() - t :.2f} sec)")

        # Define model
        label_size = 0 if self.num_classes == 1 else self.num_classes
        self.g = Generator(
            cfg.MODEL.LATENT_SIZE,
            label_size,
            cfg.RESOLUTION,
            embedding_size=cfg.MODEL.EMBEDDING_SIZE,
            dlatent_size=400,
            extra_channels=cfg.MODEL.EXTRA_CHANNEL,
            is_training=True
        ).to(self.device)

        self.d = Discriminator(
            label_size,
            cfg.RESOLUTION,
            extra_channels=cfg.MODEL.EXTRA_CHANNEL
        ).to(self.device)

        self.g_ema = copy.deepcopy(self.g).eval()

        # Define losses
        self.rec_loss = MaskedRecLoss(mask='gaussian', num_channels=1, device=self.device)

        # Define optimizers
        g_reg_ratio = cfg.TRAIN.G_REG_EVERY / (cfg.TRAIN.G_REG_EVERY + 1)
        d_reg_ratio = cfg.TRAIN.D_REG_EVERY / (cfg.TRAIN.D_REG_EVERY + 1)
        self.g_optim = optim.Adam(self.g.parameters(), lr=cfg.TRAIN.LR * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
        self.d_optim = optim.Adam(self.d.parameters(), lr=cfg.TRAIN.LR * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

        # resume from checkpoints if given
        self.start_iter = 0
        if cfg.TRAIN.NV_WEIGHTS_PATH:
            raise RuntimeError("resume from NV model is deprecated because model architecture is different")

        if cfg.TRAIN.CKPT:
            # assume checkpoint will load on device directly.
            print(f'resume model from {cfg.TRAIN.CKPT}')
            ckpt = torch.load(cfg.TRAIN.CKPT, map_location=self.device)

            self.g.load_state_dict(ckpt['g'])
            self.d.load_state_dict(ckpt['d'])
            self.g_ema.load_state_dict(ckpt['g_ema'])

            self.g_optim.load_state_dict(ckpt['g_optim'])
            self.d_optim.load_state_dict(ckpt['d_optim'])
            try:
                self.start_iter = int(Path(cfg.TRAIN.CKPT).stem.split('-')[1])
            except ValueError:
                self.log.error(f"Fail to parse #iteration from checkpoint filename. Valid format is 'ckpt-<#iter>.pt'")

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
            self.fid_tracker = FIDTracker(cfg, self.local_rank, self.num_gpus, self.out_dir, use_tqdm=(self.local_rank == 0))

    def train(self):
        cfg_d = self.cfg.DATASET
        cfg_t = self.cfg.TRAIN
        if self.start_iter >= cfg_t.ITERATION:
            print("the current iteration already meet your target iteration")
            return

        digits_length = len(str(cfg_t.ITERATION))
        ema_beta = 0.5 ** (self.batch_size * self.num_gpus / (10 * 1000))
        mean_path_length = 0
        loss_keys = ['g', 'd', 'g_rec', 'real_score', 'fake_score', 'mean_path', 'r1', 'path', 'path_length']
        loss_dict = OrderedDict((k, torch.tensor(0.0, dtype=torch.float, device=self.device)) for k in loss_keys)

        # To make state_dict consistent in mutli nodes & single node training
        g_module = self.g.module if self.ddp else self.g
        d_module = self.d.module if self.ddp else self.d

        loader = sample_data(self.loader)
        pbar = range(self.start_iter, cfg_t.ITERATION)
        if self.local_rank == 0:
            pbar = tqdm(pbar, total=cfg_t.ITERATION, initial=self.start_iter, dynamic_ncols=True, smoothing=0.01)

        # main loop
        for i in pbar:
            s = time()
            body_imgs, face_imgs, mask = [x.to(self.device) for x in next(loader)]
            masked_body = torch.cat([body_imgs * mask, mask], dim=1)

            requires_grad(self.g, False)
            requires_grad(self.d, True)

            noise = mixing_noise(self.batch_size, self.latent, cfg_t.STYLE_MIXING_PROB, self.device)
            fake_label = torch.randint(self.num_classes, (self.batch_size,)).to(self.device) if self.num_classes > 1 else None
            fake_img, _ = self.g(noise, labels_in=fake_label, style_in=face_imgs, content_in=masked_body)
            fake_pred = self.d(fake_img)
            real_pred = self.d(body_imgs)

            d_loss = logistic_loss(real_pred, fake_pred)

            loss_dict['d'] = d_loss
            loss_dict['real_score'] = real_pred.mean()
            loss_dict['fake_score'] = fake_pred.mean()

            self.d.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            if i % cfg_t.D_REG_EVERY == 0:
                body_imgs.requires_grad = True
                real_pred = self.d(body_imgs)
                r1_loss = d_r1_loss(real_pred, body_imgs)

                self.d.zero_grad()
                (cfg_t.R1 / 2 * r1_loss * cfg_t.D_REG_EVERY + 0 * real_pred[0]).backward()
                self.d_optim.step()
                loss_dict['r1'] = r1_loss

            requires_grad(self.g, True)
            requires_grad(self.d, False)

            noise = mixing_noise(self.batch_size, self.latent, cfg_t.STYLE_MIXING_PROB, self.device)
            fake_label = torch.randint(self.num_classes, (self.batch_size,)).to(self.device) if self.num_classes > 1 else None
            fake_img, _ = self.g(noise, labels_in=fake_label, style_in=face_imgs, content_in=masked_body)
            fake_pred = self.d(fake_img)
            g_loss = nonsaturating_loss(fake_pred)
            g_rec_loss = self.rec_loss(body_imgs, fake_img, mask=mask)
            loss_dict['g'] = g_loss
            loss_dict['g_rec'] = g_rec_loss

            self.g.zero_grad()
            (g_loss + g_rec_loss).backward()
            self.g_optim.step()

            if i % cfg_t.G_REG_EVERY == 0:
                self.log.debug("Apply regularization to G")
                self.g.zero_grad()
                path_batch_size = max(1, self.batch_size // cfg_t.PATH_BATCH_SHRINK)
                noise = mixing_noise(path_batch_size, self.latent, cfg_t.STYLE_MIXING_PROB, self.device)
                fake_label = (torch.randint(self.num_classes, (path_batch_size,)).to(self.device)
                              if self.num_classes > 0 else None)
                fake_img, latents = self.g(
                    noise, labels_in=fake_label, style_in=face_imgs[:path_batch_size], content_in=masked_body[:path_batch_size], return_latents=True)

                path_loss, mean_path_length, path_lengths = path_regularize(fake_img, latents, mean_path_length)
                weighted_path_loss = cfg_t.PATH_REGULARIZE * cfg_t.G_REG_EVERY * path_loss

                if cfg_t.PATH_BATCH_SHRINK:
                    weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                weighted_path_loss.backward()
                self.g_optim.step()

                loss_dict['path'] = path_loss
                loss_dict['path_length'] = path_lengths.mean()
                loss_dict['mean_path'] = mean_path_length

            accumulate(self.g_ema, g_module, ema_beta)

            if self.fid_tracker is not None and (i == 0 or (i + 1) % self.cfg.EVAL.FID.EVERY == 0):
                k_iter = (i + 1) / 1000
                self.g_ema.eval()
                self.fid_tracker.calc_fid(self.g_ema, k_iter, save=True)

            # reduce loss
            with torch.no_grad():
                losses = [torch.stack(list(loss_dict.values()), dim=0)]
                if self.num_gpus > 1:
                    torch.distributed.reduce_multigpu(losses, dst=0)

            if self.local_rank == 0:
                reduced_loss = {k: (v / self.num_gpus).item()
                                for k, v in zip(loss_dict.keys(), losses[0])}

                if i == 0 or (i + 1) % cfg_t.SAMPLE_EVERY == 0:
                    sample_iter = 'init' if i == 0 else str(i + 1).zfill(digits_length)
                    self.sampling(sample_iter)

                if (i + 1) % cfg_t.SAVE_CKPT_EVERY == 0:
                    ckpt_iter = str(i + 1).zfill(digits_length)
                    ckpt_dir = self.out_dir / 'checkpoints'
                    snapshot = {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': self.g_ema.state_dict(),
                        'g_optim': self.g_optim.state_dict(),
                        'd_optim': self.d_optim.state_dict(),
                    }
                    torch.save(snapshot, ckpt_dir / f'ckpt-{ckpt_iter}.pt')
                    ckpt_paths = list(ckpt_dir.glob('*.pt'))
                    if len(ckpt_paths) > cfg_t.CKPT_MAX_KEEP + 1:
                        ckpt_idx = sorted([int(str(p.name)[5:5 + digits_length]) for p in ckpt_paths])
                        os.remove(ckpt_dir / f'ckpt-{str(ckpt_idx[0]).zfill(digits_length)}.pt')

                if wandb and self.use_wandb:
                    wandb.log({
                        'training time': time() - s,
                        'Generator': reduced_loss['g'],
                        'G-reconstruction': reduced_loss['g_rec'],
                        'Discriminator': reduced_loss['d'],
                        'R1': reduced_loss['r1'],
                        'Path Length Regularization': reduced_loss['path'],
                        'Path Length': reduced_loss['path_length'],
                        'Mean Path Length': reduced_loss['mean_path'],
                        'Real Score': reduced_loss['real_score'],
                        'Fake Score': reduced_loss['fake_score'],
                    })

                pbar.set_description(
                    "d: {d:.4f}; g: {g:.4f}; g_rec: {g_rec:.4f}; r1: {r1:.4f}; path: {path:.4f}; mean path: {mean_path:.4f}".format(**reduced_loss)
                )

        if args.local_rank == 0 and self.fid_tracker:
            self.fid_tracker.plot_fid()

    def _get_sample_data(self):
        cfg = self.cfg
        sample = misc.EasyDict()
        sample.body_imgs, sample.face_imgs, sample.mask = [], [], []

        # ugly. concat val data from real dataset & resampling
        datasets = [get_dataset(cfg.DATASET, cfg.RESOLUTION, split='val'),
                    ResamplingDataset(cfg.DATASET, cfg.RESOLUTION)]
        for ds in datasets:
            loader = torch.utils.data.DataLoader(ds, batch_size=self.n_sample // len(datasets), shuffle=False, num_workers=0)
            body_imgs, face_imgs, mask = [x.to(self.device) for x in next(iter(loader))]
            sample.body_imgs.append(body_imgs)
            sample.face_imgs.append(face_imgs)
            sample.mask.append(mask)
            del loader

        sample.body_imgs = torch.cat(sample.body_imgs, dim=0)
        sample.face_imgs = torch.cat(sample.face_imgs, dim=0)
        sample.mask = torch.cat(sample.mask, dim=0)
        sample.masked_body = torch.cat([sample.body_imgs * sample.mask, sample.mask], dim=1)
        sample.z = torch.randn(self.n_sample, self.latent, device=self.device)
        sample.label = torch.randint(self.num_classes, (sample.z.shape[0],)).to(self.device) if self.num_classes > 1 else None
        self.log.debug(f"sample vector: {sample.z.shape}")

        return sample

    def sampling(self, idx):
        assert self.local_rank == 0
        if self.sample is None:
            self.sample = self._get_sample_data()

        cfg = self.cfg.DATASET
        with torch.no_grad():
            self.g_ema.eval()
            samples, _ = self.g_ema([self.sample.z], labels_in=self.sample.label, style_in=self.sample.face_imgs, content_in=self.sample.masked_body)
            b, c, h, w = samples.shape
            if cfg.DATASET == 'MultiChannelDataset':
                assert c == sum(cfg.CHANNELS)
                samples = torch.split(samples, cfg.CHANNELS, dim=1)
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
                range=(-1, 1),  # value_range in PyTorch 1.8+
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='torch.distributed.launch')
    parser.add_argument('-c', '--cfg', help="path to the configuration file", metavar='PATH')
    parser.add_argument('-o', '--out_dir', metavar='PATH',
                        help="path to output directory. If not set, auto. set to subdirectory of OUT_DIR in configuration")
    parser.add_argument('--local_rank', type=int, default=0, metavar='INT', help='Automatically given by %(prog)s')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb', action='store_true')
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
        misc.prepare_training(args, cfg)
        print(cfg)
        shutil.copy(args.cfg, args.out_dir)

    logger = misc.create_logger(**vars(args))

    t = time()
    logger.info("initialize trainer...")
    trainer = Trainer(args, cfg, logger)
    logger.info(f"trainer initialized. ({time() - t :.2f} sec)")

    if args.local_rank == 0 and args.wandb:
        logger.info(f"initialize wandb project: {Path(args.cfg).stem}")
        wandb.init(
            project=f'stylegan2-{Path(args.cfg).stem}',
            config=convert_to_dict(cfg)
        )

    cfg.freeze()
    trainer.train()

    if args.local_rank == 0:
        (args.out_dir / 'finish.txt').touch()
