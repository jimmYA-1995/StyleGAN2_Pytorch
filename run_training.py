import argparse
import math
import random
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3" ###
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from models import Generator, Discriminator
from losses import nonsaturating_loss, path_regularize, logistic_loss, d_r1_loss
from flags import get_arguments
from dataset import MultiResolutionDataset, ImageFolderDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def get_result_dir(root, n_gpu, dataset):
    p = Path(root)
    run_id = '00000'
    experiment = 'train'
    dataset_name = dataset.split('/')[-1]
    if not p.exists():
        p.mkdir()

    ids = [int(str(x).split('/')[-1][:5]) for x in p.iterdir()]
    if len(ids) > 0:
        run_id = str(sorted(ids)[-1] + 1).zfill(5)
    
    result_dir = p / f'{run_id}-{experiment}-{n_gpu}gpu-{dataset_name}'
    result_dir.mkdir()
    return result_dir

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def get_dataloader(data_path, resolution, distributed=True):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    
    if "mpii" in data_path:
        print("using ImageFolder dataset")
        dataset = ImageFolderDataset(data_path, transform, resolution)
    else:
        dataset = MultiResolutionDataset(data_path, transform, resolution)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=distributed),
        drop_last=True,
    )
    return loader

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


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


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

class Trainer():
    def __init__(self, args):
        
        # dummy. mod. in the future
        self.device = 'cuda'
        self.local_rank = args.local_rank
        self.distributed = args.distributed
        self.result_dir = args.result_dir
        self.n_sample = args.n_sample
        self.latent = args.latent
        self.batch = args.batch
        self.mixing = args.mixing
        self.r1 = args.r1
        self.g_reg_every = args.g_reg_every
        self.d_reg_every = args.d_reg_every
        self.path_regularize = args.path_regularize
        self.path_batch_shrink = args.path_batch_shrink
        self.use_wandb = args.wandb
        
        # datset
        self.loader = get_dataloader(args.path, args.size, args.distributed)

        # Define model
        self.generator = Generator(args.latent, 0, args.size, is_training=True).to(self.device)
        self.discriminator = Discriminator(0, args.size).to(self.device)
        self.g_ema = Generator(args.latent, 0, args.size, is_training=False).to(self.device)    
        self.g_ema.eval()
        accumulate(self.g_ema, self.generator, 0)
        
        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )
        self.d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )
        
        self.total_step = args.iter
        self.start_iter = 0
        if args.ckpt is not None:
            print('load model:', args.ckpt)
            ckpt = torch.load(args.ckpt)

            try:
                ckpt_name = os.path.basename(args.ckpt)
                self.start_iter = int(os.path.splitext(ckpt_name)[0])
            except ValueError:
                pass

            self.generator.load_state_dict(ckpt['g'])
            self.discriminator.load_state_dict(ckpt['d'])
            self.g_ema.load_state_dict(ckpt['g_ema'])

            self.g_optim.load_state_dict(ckpt['g_optim'])
            self.d_optim.load_state_dict(ckpt['d_optim'])

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

    def train(self):
        loader = sample_data(self.loader)
        
        if self.start_iter >= self.total_step:
            print("the current iteration already meet your target iteration")
            return

        pbar = range(self.start_iter, self.total_step)
        if get_rank() == 0:
            pbar = tqdm(pbar, total=self.total_step, initial=self.start_iter, dynamic_ncols=True, smoothing=0.01)

        # init. all 
        mean_path_length = 0
        g_loss_val = 0
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

        sample_z = torch.randn(self.n_sample, self.latent, device=self.device)

        # start training
        for i in pbar:
            real_img = next(loader)[0] ###
            real_img = real_img.to(self.device)

            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)

            noise = mixing_noise(self.batch, self.latent, self.mixing, self.device)
            fake_img, _ = self.generator(noise)
            fake_pred = self.discriminator(fake_img)
            real_pred = self.discriminator(real_img)

            d_loss = logistic_loss(real_pred, fake_pred)

            loss_dict['d'] = d_loss
            loss_dict['real_score'] = real_pred.mean()
            loss_dict['fake_score'] = fake_pred.mean()

            self.discriminator.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            d_regularize = i % self.d_reg_every == 0
            if d_regularize:
                real_img.requires_grad = True
                real_pred = self.discriminator(real_img)
                r1_loss = d_r1_loss(real_pred, real_img)

                self.discriminator.zero_grad()
                (self.r1 / 2 * r1_loss * self.d_reg_every + 0 * real_pred[0]).backward()
                self.d_optim.step()

            loss_dict['r1'] = r1_loss

            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)

            noise = mixing_noise(self.batch, self.latent, self.mixing, self.device)
            fake_img, _ = self.generator(noise)
            fake_pred = self.discriminator(fake_img)
            g_loss = nonsaturating_loss(fake_pred)

            loss_dict['g'] = g_loss

            self.generator.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            g_regularize = i % self.g_reg_every == 0

            if g_regularize:
                path_batch_size = max(1, self.batch // self.path_batch_shrink)
                noise = mixing_noise(
                    path_batch_size, self.latent, self.mixing, self.device
                )
                fake_img, latents = self.generator(noise, return_latents=True)

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

            loss_reduced = reduce_loss_dict(loss_dict)

            d_loss_val = loss_reduced['d'].mean().item()
            g_loss_val = loss_reduced['g'].mean().item()
            r1_val = loss_reduced['r1'].mean().item()
            path_loss_val = loss_reduced['path'].mean().item()
            real_score_val = loss_reduced['real_score'].mean().item()
            fake_score_val = loss_reduced['fake_score'].mean().item()
            path_length_val = loss_reduced['path_length'].mean().item()

            if get_rank() == 0:
                pbar.set_description(
                    (
                        f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; '
                        f'path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}'
                    )
                )

                if wandb and self.use_wandb:
                    wandb.log(
                        {
                            'Generator': g_loss_val,
                            'Discriminator': d_loss_val,
                            'R1': r1_val,
                            'Path Length Regularization': path_loss_val,
                            'Mean Path Length': mean_path_length,
                            'Real Score': real_score_val,
                            'Fake Score': fake_score_val,
                            'Path Length': path_length_val,
                        }
                    )

                if i % 100 == 0:
                    with torch.no_grad():
                        self.g_ema.eval()
                        sample, _ = self.g_ema([sample_z])
                        utils.save_image(
                            sample,
                            self.result_dir / f'sample-{str(i).zfill(6)}.png',
                            nrow=int(self.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                if i % 10000 == 0:
                    torch.save(
                        {
                            'g': g_module.state_dict(),
                            'd': d_module.state_dict(),
                            'g_ema': self.g_ema.state_dict(),
                            'g_optim': self.g_optim.state_dict(),
                            'd_optim': self.d_optim.state_dict(),
                        },
                        self.result_dir / f'ckpt-{str(i).zfill(6)}.pt',
                    )


if __name__ == '__main__':

    args = get_arguments()
    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = False if args.local or n_gpu <= 1 else True
    
    
    args.result_dir = get_result_dir(args.result_dir, n_gpu, args.path)
    with open(args.result_dir / 'log.txt', 'w') as f:
        print(vars(args), file=f)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    trainer = Trainer(args)

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project='stylegan2_mpii')
        
    trainer.train()
