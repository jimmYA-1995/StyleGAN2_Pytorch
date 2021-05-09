import os
import sys
import random
import argparse
from time import time
from pathlib import Path
from pprint import pformat

import wandb
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torchvision import utils

import misc
from config import get_cfg_defaults, convert_to_dict
from dataset import get_dataset, get_dataloader
from models import Generator, Discriminator
from models.utils import load_weights_from_nv, load_partial_weights
from losses import nonsaturating_loss, path_regularize, logistic_loss, d_r1_loss, masked_l1_loss
from metrics import FIDTracker
from distributed import master_only, get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size


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
        self.device = 'cuda'
        self.cfg = cfg
        self.log = logger
        self.local_rank = args.local_rank
        self.distributed = args.num_gpus > 1
        self.out_dir = args.out_dir
        self.use_wandb = args.wandb
        self.n_sample = cfg.N_SAMPLE
        self.resolution = cfg.RESOLUTION
        self.num_classes = cfg.N_CLASSES

        # model
        self.n_mlp = cfg.MODEL.N_MLP
        self.latent = cfg.MODEL.LATENT_SIZE
        self.embed_size = cfg.MODEL.EMBEDDING_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        self.mixing = cfg.TRAIN.STYLE_MIXING_PROB
        self.r1 = cfg.TRAIN.R1
        self.g_reg_every = cfg.TRAIN.G_REG_EVERY
        self.d_reg_every = cfg.TRAIN.D_REG_EVERY
        self.path_regularize = cfg.TRAIN.PATH_REGULARIZE
        self.path_batch_shrink = cfg.TRAIN.PATH_BATCH_SHRINK

        self.fid_tracker = None

        # datset
        t = time()
        print("get dataloader ...", end='\r')
        self.loader = get_dataloader(cfg, self.batch_size, distributed=self.distributed)
        self.val_loader = get_dataloader(cfg, self.n_sample, split='val', distributed=self.distributed)
        print(f"get dataloader complete ({time() - t :.2f} sec)")

        # Define model
        label_size = 0 if self.num_classes == 1 else self.num_classes
        self.generator = Generator(
            self.latent,
            label_size,
            self.resolution,
            embedding_size=self.embed_size,
            dlatents_size=256,
            extra_channels=cfg.MODEL.EXTRA_CHANNEL,
            is_training=True
        ).to(self.device)

        self.discriminator = Discriminator(
            label_size,
            self.resolution,
            extra_channels=cfg.MODEL.EXTRA_CHANNEL
        ).to(self.device)

        self.g_ema = Generator(
            self.latent,
            label_size,
            self.resolution,
            embedding_size=self.embed_size,
            dlatents_size=256,
            extra_channels=cfg.MODEL.EXTRA_CHANNEL,
            is_training=False
        ).to(self.device)
        self.g_ema.eval()
        accumulate(self.g_ema, self.generator, 0)

        g_reg_ratio = self.g_reg_every / (self.g_reg_every + 1)
        d_reg_ratio = self.d_reg_every / (self.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.generator.parameters(),
            lr=cfg.TRAIN.LR * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )
        self.d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.TRAIN.LR * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )

        self.total_step = cfg.TRAIN.ITERATION
        self.start_iter = 0
        if cfg.TRAIN.NV_WEIGHTS_PATH:
            load_weights_from_nv(
                self.generator, self.discriminator, self.g_ema, cfg.TRAIN.NV_WEIGHTS_PATH)
        elif cfg.TRAIN.CKPT:
            print(f'resume model from {cfg.TRAIN.CKPT}')
            ckpt = torch.load(cfg.TRAIN.CKPT)

            try:
                self.start_iter = int(Path(cfg.TRAIN.CKPT).stem.split('-')[1])
            except ValueError:
                logger.info('**** load ckpt failed. start from scratch ****')
                pass

            try:
                self.generator.load_state_dict(ckpt['g'])
                self.discriminator.load_state_dict(ckpt['d'])
                self.g_ema.load_state_dict(ckpt['g_ema'])

                self.g_optim.load_state_dict(ckpt['g_optim'])
                self.d_optim.load_state_dict(ckpt['d_optim'])
            except RuntimeError:
                logger.warn(
                    " *** using hacky way to load partial weight to model *** ")
                self.start_iter = load_partial_weights(
                    self.generator, self.discriminator, self.g_ema, ckpt, logger=logger)

        if self.distributed:
            self.generator = nn.parallel.DistributedDataParallel(
                self.generator,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False
            )

            self.discriminator = nn.parallel.DistributedDataParallel(
                self.discriminator,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False
            )

        # init. FID tracker if needed.
        if 'fid' in cfg.EVAL.METRICS.split(',') and get_rank() == 0:
            self.fid_tracker = FIDTracker(cfg, self.out_dir, use_tqdm=True)
        synchronize()

    def train(self):
        cfg_d = self.cfg.DATASET
        cfg_t = self.cfg.TRAIN

        digits_length = len(str(cfg_t.ITERATION))
        loader = sample_data(self.loader)

        if self.start_iter >= self.total_step:
            print("the current iteration already meet your target iteration")
            return

        pbar = range(self.start_iter, self.total_step)
        if get_rank() == 0:
            pbar = tqdm(pbar, total=self.total_step,
                        initial=self.start_iter, dynamic_ncols=True, smoothing=0.01)

        # init. all
        mean_path_length = 0
        g_loss_val = 0
        g_rec_loss_val = 0
        d_loss_val = 0
        r1_loss = torch.tensor(0.0, device=self.device)
        path_loss = torch.tensor(0.0, device=self.device)
        path_lengths = torch.tensor(0.0, device=self.device)
        mean_path_length_avg = 0
        loss_dict = {}

        if self.distributed:
            g_module = self.generator.module
            d_module = self.discriminator.module
        else:
            g_module = self.generator
            d_module = self.discriminator

        accum = 0.5 ** (32 / (10 * 1000))

        val_loader = sample_data(self.val_loader)
        sample_body_imgs, sample_face_imgs, sample_mask = [
            x.to(self.device) for x in next(val_loader)]
        sample_masked_body = sample_body_imgs * sample_mask
        del val_loader
        sample_z = torch.randn(self.n_sample, self.latent, device=self.device)
        sample_label = (torch.randint(self.num_classes, (sample_z.shape[0],)).to(self.device)
                        if self.num_classes > 1 else None)
        self.log.debug(f"sample vector: {sample_z.shape}")

        # main loop
        for i in pbar:
            s = time()
            body_imgs, face_imgs, mask = [
                x.to(self.device) for x in next(loader)]
            masked_body = body_imgs * mask

            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)

            noise = mixing_noise(
                self.batch_size, self.latent, self.mixing, self.device)
            fake_label = (torch.randint(self.num_classes, (self.batch_size,)).to(self.device)
                          if self.num_classes > 1 else None)
            fake_img, _ = self.generator(
                noise, labels_in=fake_label, style_in=face_imgs, content_in=masked_body)
            fake_pred = self.discriminator(fake_img)
            real_pred = self.discriminator(body_imgs)

            d_loss = logistic_loss(real_pred, fake_pred)

            loss_dict['d'] = d_loss
            loss_dict['real_score'] = real_pred.mean()
            loss_dict['fake_score'] = fake_pred.mean()

            self.discriminator.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            d_regularize = i % self.d_reg_every == 0
            if d_regularize:
                body_imgs.requires_grad = True
                real_pred = self.discriminator(body_imgs)
                r1_loss = d_r1_loss(real_pred, body_imgs)

                self.discriminator.zero_grad()
                (self.r1 / 2 * r1_loss * self.d_reg_every + 0 * real_pred[0]).backward()
                self.d_optim.step()

            loss_dict['r1'] = r1_loss

            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)

            noise = mixing_noise(
                self.batch_size, self.latent, self.mixing, self.device)
            fake_label = (torch.randint(self.num_classes, (self.batch_size,)).to(self.device)
                          if self.num_classes > 1 else None)
            fake_img, _ = self.generator(
                noise, labels_in=fake_label, style_in=face_imgs, content_in=masked_body)
            fake_pred = self.discriminator(fake_img)
            g_loss = nonsaturating_loss(fake_pred)
            g_rec_loss = masked_l1_loss(
                masked_body, fake_img, mask=mask) * 256 * 256
            loss_dict['g'] = g_loss
            loss_dict['g_rec'] = g_rec_loss

            self.generator.zero_grad()
            (g_loss + g_rec_loss).backward()
            self.g_optim.step()

            g_regularize = i % self.g_reg_every == 0

            if g_regularize:
                self.log.debug("Apply regularization to G")
                path_batch_size = max(
                    1, self.batch_size // self.path_batch_shrink)
                noise = mixing_noise(
                    path_batch_size, self.latent, self.mixing, self.device
                )
                fake_label = (torch.randint(self.num_classes, (path_batch_size,)).to(self.device)
                              if self.num_classes > 0 else None)
                fake_img, latents = self.generator(
                    noise, labels_in=fake_label, style_in=face_imgs[:path_batch_size], content_in=masked_body[:path_batch_size], return_latents=True)

                path_loss, mean_path_length, path_lengths = path_regularize(
                    fake_img, latents, mean_path_length
                )

                self.generator.zero_grad()
                weighted_path_loss = self.path_regularize * self.g_reg_every * path_loss

                if self.path_batch_shrink:
                    weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                weighted_path_loss.backward()

                self.g_optim.step()

                mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
                )

            loss_dict['path'] = path_loss
            loss_dict['path_length'] = path_lengths.mean()

            accumulate(self.g_ema, g_module, accum)

            if (get_rank() == 0 and i != 0 and (i + 1) % self.cfg.EVAL.FID.EVERY == 0):
                k_iter = (i + 1) / 1000
                self.g_ema.eval()
                self.fid_tracker.calc_fid(self.g_ema, k_iter, save=True)

            loss_reduced = reduce_loss_dict(loss_dict)

            d_loss_val = loss_reduced['d'].mean().item()
            g_loss_val = loss_reduced['g'].mean().item()
            g_rec_loss_val = loss_reduced['g_rec'].mean().item()
            r1_val = loss_reduced['r1'].mean().item()
            path_loss_val = loss_reduced['path'].mean().item()
            real_score_val = loss_reduced['real_score'].mean().item()
            fake_score_val = loss_reduced['fake_score'].mean().item()
            path_length_val = loss_reduced['path_length'].mean().item()

            if get_rank() == 0:
                pbar.set_description(
                    (
                        f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; g_rec: {g_rec_loss_val:.4f}; r1: {r1_val:.4f}; '
                        f'path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}'
                    )
                )

                if i == 0 or (i + 1) % cfg_t.SAMPLE_EVERY == 0:
                    sample_iter = 'init' if i == 0 else str(
                        i).zfill(digits_length)
                    with torch.no_grad():
                        self.g_ema.eval()
                        sample, _ = self.g_ema(
                            [sample_z], labels_in=sample_label, style_in=sample_face_imgs, content_in=sample_masked_body)

                        if cfg_d.DATASET == 'MultiChannelDataset':
                            sample_list = torch.split(sample, cfg_d.CHANNELS, dim=1)
                            sample_list = [(x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x)
                                           for x in sample_list]

                            utils.save_image(
                                list(torch.cat(sample_list, dim=2).unbind(0)),
                                self.out_dir / f'samples/fake-{sample_iter}.png',
                                nrow=int(self.n_sample ** 0.5) * 3,
                                normalize=True,
                                range=(-1, 1),  # value_range in PyTorch 1.8+
                            )

                        else:
                            b, c, h, w = sample.shape
                            stack_samples = torch.stack(
                                [sample_face_imgs, sample_body_imgs, sample], dim=0)
                            samples = torch.transpose(
                                stack_samples, 0, 1).reshape(3 * b, c, h, w)
                            utils.save_image(
                                samples,
                                self.out_dir / f'samples/fake-{sample_iter}.png',
                                nrow=int(self.n_sample ** 0.5) * 3,
                                normalize=True,
                                range=(-1, 1),
                            )

                if i == 0 or (i + 1) % cfg_t.SAVE_CKPT_EVERY == 0:
                    ckpt_iter = str(i + 1).zfill(digits_length)
                    snapshot = {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': self.g_ema.state_dict(),
                        'g_optim': self.g_optim.state_dict(),
                        'd_optim': self.d_optim.state_dict(),
                    }
                    torch.save(snapshot, self.out_dir / f'checkpoints/ckpt-{ckpt_iter}.pt')
                    ckpt_dir = self.out_dir / 'checkpoints'
                    ckpt_paths = list(ckpt_dir.glob('*.pt'))
                    if len(ckpt_paths) > cfg_t.CKPT_MAX_KEEP + 1:
                        ckpt_idx = sorted(
                            [int(str(p.name)[5:5 + digits_length]) for p in ckpt_paths])
                        os.remove(
                            ckpt_dir / f'ckpt-{str(ckpt_idx[1]).zfill(digits_length)}.pt')

                if wandb and self.use_wandb:
                    wandb.log({
                        'training time': time() - s,
                        'Generator': g_loss_val,
                        'G-reconstruction': g_rec_loss_val,
                        'Discriminator': d_loss_val,
                        'R1': r1_val,
                        'Path Length Regularization': path_loss_val,
                        'Mean Path Length': mean_path_length,
                        'Real Score': real_score_val,
                        'Fake Score': fake_score_val,
                        'Path Length': path_length_val,
                    })

        if args.local_rank == 0 and self.fid_tracker:
            self.fid_tracker.plot_fid()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='torch.distributed.launch')
    parser.add_argument('-c', '--cfg', help="path to the configuration file", metavar='PATH')
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
        synchronize()

        print = master_only(print)
        print("print function overriden")

    if args.num_gpus == 1 or args.local_rank == 0:
        misc.prepare_training(args, cfg)

    logger = misc.create_logger(args.out_dir, args.local_rank, args.debug)

    logger.debug(pformat(cfg))
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
