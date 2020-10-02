import os
import sys
import math
import random
import logging
from time import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn, autograd, optim
from torch.nn import functional as F
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from dataset import get_dataset, get_dataloader
from misc import parse_args, prepare_training
from load_weights import load_weights_from_nv
from models import Generator, Discriminator
from losses import nonsaturating_loss, path_regularize, logistic_loss, d_r1_loss
from metrics import FIDTracker
from config import config, update_config
from distributed import (
    master_only,
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


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
    def __init__(self, args, config, logger):
        
        # dummy. mod. in the future
        self.device = 'cuda'
        self.config = config
        self.logger = logger
        self.local_rank = args.local_rank
        self.distributed = args.distributed
        self.out_dir = args.out_dir
        self.use_wandb = args.wandb
        self.n_sample = config.N_SAMPLE
        self.resolution = config.RESOLUTION
        self.num_classes = config.N_CLASSES
        # model
        
        self.n_mlp = config.MODEL.N_MLP
        self.latent = config.MODEL.LATENT_SIZE
        self.embed_size = config.MODEL.EMBEDDING_SIZE
        # self.extra_channels = config.MODEL.EXTRA_CHANNEL
        self.batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
        self.mixing = config.TRAIN.STYLE_MIXING_PROB
        self.r1 = config.TRAIN.R1
        self.g_reg_every = config.TRAIN.G_REG_EVERY
        self.d_reg_every = config.TRAIN.D_REG_EVERY
        self.path_regularize = config.TRAIN.PATH_REGULARIZE
        self.path_batch_shrink = config.TRAIN.PATH_BATCH_SHRINK
        
        self.fid_tracker = None
        
        # datset
        print("get dataloader ...")
        t = time()
        self.loader = get_dataloader(config, self.batch_size, distributed=args.distributed)
        print(f"get dataloader complete ({time() - t})")
        
        self.use_sk = False
        self.use_mk = False
        if config.MODEL.EXTRA_CHANNEL > 0:
            if 'skeleton' in config.DATASET.SOURCE[1]:
                self.use_sk = True
            if 'mask' in config.DATASET.SOURCE[-1]:
                self.use_mk = True
        # Define model
        self.generator = Generator(self.latent, self.num_classes, self.resolution,
                                   embedding_size=self.embed_size, extra_channels=config.MODEL.EXTRA_CHANNEL,
                                   use_sk=self.use_sk, use_mk=self.use_mk, is_training=True).to(self.device)
        self.discriminator = Discriminator(0, self.resolution, extra_channels=config.MODEL.EXTRA_CHANNEL).to(self.device)
        self.g_ema = Generator(self.latent, self.num_classes, self.resolution,
                               embedding_size=self.embed_size, extra_channels=config.MODEL.EXTRA_CHANNEL,
                               use_sk=self.use_sk, use_mk=self.use_mk, is_training=False).to(self.device)
        self.g_ema.eval()
        accumulate(self.g_ema, self.generator, 0)
        
        g_reg_ratio = self.g_reg_every / (self.g_reg_every + 1)
        d_reg_ratio = self.d_reg_every / (self.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.generator.parameters(),
            lr=config.TRAIN.LR * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )
        self.d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=config.TRAIN.LR * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )
        
        self.total_step = config.TRAIN.ITERATION
        self.start_iter = 0
        if config.TRAIN.NV_WEIGHTS_PATH:
            load_weights_from_nv(self.generator, self.discriminator, self.g_ema, config.TRAIN.NV_WEIGHTS_PATH)
        elif config.TRAIN.CKPT:
            print('load model:', config.TRAIN.CKPT)
            ckpt = torch.load(config.TRAIN.CKPT)

            try:
                self.start_iter = int(Path(config.TRAIN.CKPT).stem.split('-')[1])
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
                logger.warn(" ******* using hacky way to load partial weight to model ******* ")
                try:
                    for k,v in self.generator.named_parameters():
                        if 'trgb' in k:
                            if 'conv.w' in k:
                                with torch.no_grad():
                                    v[:3, ...].copy_(ckpt['g'][k])
                            elif 'bias' in k:
                                with torch.no_grad():
                                    v[:, :3, ...].copy_(ckpt['g'][k])
                            else:
                                with torch.no_grad():
                                    v.copy_(ckpt['g'][k])
                        else:
                            with torch.no_grad():
                                v.copy_(ckpt['g'][k])
                            v.requires_grad = False
                                
                    for k,v in self.discriminator.named_parameters():
                        if 'frgb' in k:
                            if 'conv.w' in k:
                                with torch.no_grad():
                                    v[:, :3, ...].copy_(ckpt['d'][k])
                        else:
                            with torch.no_grad():
                                v.copy_(ckpt['d'][k])
                            v.requires_grad = False
                                
                    for k,v in self.g_ema.named_parameters():
                        if 'trgb' in k:
                            if 'conv.w' in k:
                                with torch.no_grad():
                                    v[:3, ...].copy_(ckpt['g_ema'][k])
                            elif 'bias' in k:
                                with torch.no_grad():
                                    v[:, :3, ...].copy_(ckpt['g_ema'][k])
                        else:
                            with torch.no_grad():
                                v.copy_(ckpt['g_ema'][k])
                    logger.info("Transfer learning. Set start iteration to 0")
                    self.start_iter = 0
                except RuntimeError:
                    logger.error(" ***** fail to load partial weights to models ***** ")
                     

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
        if 'fid' in config.EVAL.METRICS.split(','):
            if get_rank() == 0:
                self.fid_tracker = FIDTracker(config, self.out_dir, use_tqdm=True)
            synchronize()
            
    def train(self):
        cfg_d = self.config.DATASET
        cfg_t = self.config.TRAIN
        
        digits_length = len(str(cfg_t.ITERATION))
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
        fixed_fake_label = torch.randint(self.num_classes, (sample_z.shape[0],)).to(self.device) \
                           if self.num_classes > 0 else None
        self.logger.debug(f"sample vector: {sample_z.shape}")
        sample_sk, sample_mk = None, None
        if self.use_sk:
            import skimage.io as io
            sample_sk = []
                
            trf = [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [5],
                                             inplace=True),
                    ]
            transform = transforms.Compose(trf)
            
            for i, p in enumerate((Path(config.DATASET.ROOTS[0]) / 'sample_sk').glob('*.jpg')):
                if i == self.n_sample: 
                    break
                sample_sk.append(transform(io.imread(p)[..., None])[None, ...])
            sample_sk = torch.cat(sample_sk, dim=0).to('cuda')
            

            self.logger.debug(f"sample sk: {sample_sk.shape}, {sample_sk.max()}, {sample_sk.min()}, {type(sample_sk)}")
        
        # start training
        for i in pbar:
            s = time()
            real_img, labels = next(loader)
            real_img = real_img.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            self.logger.debug(f"input shape: {real_img.shape}")
            real_sk = real_mk = None
            if not self.use_sk:
                real_img = real_img[:, :3, ...]
            else:
                if real_img.shape[1] == 4:
                    real_sk = real_img[:, 3:, ...]
                    real_img = real_img[:, :3, ...]
                elif real_img.shape[1] == 5:
                    real_mk = real_img[:, 4:, ...]
                    real_sk = real_img[:, 3:4, ...]
                    real_img = real_img[:, :3, ...]
                self.logger.debug(f"[sk] type: {type(real_sk)} shape: {real_sk.shape}")
                self.logger.debug(f"[sk] mean: {real_sk.mean()} std: {real_sk.std()} min:{real_sk.min()} max:{real_sk.max()}")
            
            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)

            noise = mixing_noise(self.batch_size, self.latent, self.mixing, self.device)
            fake_label = torch.randint(self.num_classes, (self.batch_size,)).to(self.device) \
                         if self.num_classes > 0 else None
            fake_img, _ = self.generator(noise, labels_in=fake_label, sk=real_sk, mk=real_mk)
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

            noise = mixing_noise(self.batch_size, self.latent, self.mixing, self.device)
            fake_label = torch.randint(self.num_classes, (self.batch_size,)).to(self.device) \
                         if self.num_classes > 0 else None
            fake_img, _ = self.generator(noise, labels_in=fake_label, sk=real_sk, mk=real_mk)
            fake_pred = self.discriminator(fake_img)
            g_loss = nonsaturating_loss(fake_pred)

            loss_dict['g'] = g_loss

            self.generator.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            g_regularize = i % self.g_reg_every == 0

            if g_regularize:
                self.logger.debug("Apply regularization to G")
                path_batch_size = max(1, self.batch_size // self.path_batch_shrink)
                noise = mixing_noise(
                    path_batch_size, self.latent, self.mixing, self.device
                )
                real_sk_reg = real_sk[:path_batch_size] if real_sk is not None else None
                real_mk_reg = real_mk[:path_batch_size] if real_mk is not None else None
                fake_label = torch.randint(self.num_classes, (path_batch_size,)).to(self.device) \
                             if self.num_classes > 0 else None
                fake_img, latents = self.generator(noise, labels_in=fake_label, sk=real_sk_reg, mk=real_mk_reg, return_latents=True)

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
            
            if get_rank() == 0 and self.fid_tracker is not None and \
                (i == 0 or (i+1) % self.config.EVAL.FID.EVERY == 0):
                k_iter = (i+1) / 1000
                self.g_ema.eval()
                self.fid_tracker.calc_fid(self.g_ema, k_iter, save=True)

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

                if i == 0 or (i+1) % cfg_t.SAMPLE_EVERY == 0:
                    sample_iter = 'init' if i==0 else str(i).zfill(digits_length)
                    with torch.no_grad():
                        self.g_ema.eval()
                        sample, _ = self.g_ema([sample_z], labels_in=fixed_fake_label, sk=sample_sk, mk=sample_mk)
                        
                        if cfg_d.DATASET == 'MultiChannelDataset' and not self.use_sk:
                            s = 0
                            samples = []
                            for i, src in enumerate(cfg_d.SOURCE):
                                e = s + cfg_d.CHANNELS[i]
                                
                                _sample = sample[:,s:e,:,:]
                                if cfg_d.CHANNELS[i] == 1:
                                    _sample = _sample.repeat((1,3,1,1))
                                    
                                samples.append(_sample)
                                s = e
                                
                            utils.save_image(
                                list(torch.cat(samples, dim=2).unbind(0)),
                                self.out_dir / f'samples/fake-{sample_iter}.png',
                                nrow=int(self.n_sample ** 0.5) * 3,
                                normalize=True,
                                range=(-1, 1),
                            )

                        else:
                            utils.save_image(
                                sample,
                                self.out_dir / f'samples/fake-{sample_iter}.png',
                                nrow=int(self.n_sample ** 0.5),
                                normalize=True,
                                range=(-1, 1),
                            )

                if i == 0 or (i+1) % cfg_t.SAVE_CKPT_EVERY == 0:
                    ckpt_iter = str(i+1).zfill(digits_length)
                    torch.save(
                        {
                            'g': g_module.state_dict(),
                            'd': d_module.state_dict(),
                            'g_ema': self.g_ema.state_dict(),
                            'g_optim': self.g_optim.state_dict(),
                            'd_optim': self.d_optim.state_dict(),
                        },
                        self.out_dir /  f'checkpoints/ckpt-{ckpt_iter}.pt',
                    )
                    ckpt_dir = self.out_dir / 'checkpoints'
                    ckpt_paths = list(ckpt_dir.glob('*.pt'))
                    if len(ckpt_paths) > cfg_t.CKPT_MAX_KEEP+1:
                        ckpt_idx = sorted([int(str(p.name)[5:5+digits_length]) \
                                           for p in ckpt_paths])
                        os.remove(ckpt_dir / f'ckpt-{str(ckpt_idx[1]).zfill(digits_length)}.pt')

                if wandb and self.use_wandb:
                    wandb.log(
                        {
                            'training time': time() - s,
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


if __name__ == '__main__':
    args = parse_args()
    update_config(config, args)
    
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print("CUDA VISIBLE DEVICES: ", os.environ['CUDA_VISIBLE_DEVICES'])
    n_gpu = torch.cuda.device_count()
    if args.local_rank >= n_gpu:
        raise RuntimeError('Recommend one process per device')
    args.distributed = n_gpu > 1
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()
        
        # maybe use configuration file to control master-slave behavior
        print = master_only(print)
        prepare_training = master_only(prepare_training)
        print("print function overriden")
    
    logger, args.out_dir = prepare_training(config, args.cfg, debug=args.debug)
    
    
    if not logger:
        # add process info to slave process
        log_format = '<SLAVE{}> %(levelname)-8s %(asctime)-15s %(message)s'.format(get_rank())
        logging.basicConfig(level=logging.WARN,
                            format=log_format)
        logger = logging.getLogger()

    logger.info("Only keep logs of master in log file")
    
    logger.info("initialize trainer...")
    t = time()
    trainer = Trainer(args, config, logger)
    logger.info(f"trainer initialized. (costs {time() - t})")

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project='stylegan2-appendSK')
    
    logger.info("start training")
    trainer.train()
    
    if get_rank() == 0 and getattr(trainer, 'fid_tracker') is not None:
        trainer.fid_tracker.plot_fid()
        (args.out_dir / 'finish.txt').touch()
    