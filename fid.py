import argparse
import logging
import pickle

from pathlib import Path

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import Generator
from calc_inception import load_patched_inception_v3
from misc import parse_args
from config import config, update_config
from run_training import get_dataloader

@torch.no_grad()
def extract_feature_from_real_images(config, args, inception, device):
    
    def sample_data(loader):
        while True:
            for batch in loader:
                yield batch
    
    print("getting dataloader of real images...")
    loader = get_dataloader(config, args=args, distributed=False)
    loader = sample_data(loader)
    print('loading dataloader complete')
    n_batch = args.n_sample // args.batch
    resid = args.n_sample % args.batch
    
    features = []

    print('extract features from real images...')
    for i in tqdm(range(n_batch + 1)):
        batch_size = resid if i==n_batch else args.batch
        if batch_size == 0:
            continue
        imgs = next(loader)[0][:batch_size, :3, :, :].to(device) ###
        feat = inception(imgs)[0].view(batch_size, -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0).numpy()
    print(f'complete. total extracted features: {features.shape[0]}')
    
    real_mean = np.mean(features, 0)
    real_cov = np.cov(features, rowvar=False)
    return real_mean, real_cov



@torch.no_grad()
def extract_feature_from_ckpt(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device
):
    n_batch = n_sample // batch_size
    resid = n_sample % batch_size
    features = []
    
    for i in tqdm(range(n_batch+1)):
        batch = resid if i==n_batch else batch_size
        if batch==0:
            continue
        latent = torch.randn(batch, 512, device=device)
        img, _ = generator([latent]) ## , truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img[:, :3, :, :])[0].view(img.shape[0], -1) ## fix to skeleton_channel = 3
        features.append(feat.to('cpu'))
    
    features = torch.cat(features, 0)
    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


def fid(config, args):
    if args.ckpt.is_dir():
        logging.info(f'calculating ckpt in directory {str(args.ckpt)}')
        ckpts = sorted(list(args.ckpt.glob('*.pt')))
        file_path = args.ckpt / 'fid.txt'
        plot_path = args.ckpt / 'fid.png'
    elif args.ckpt.is_file():
        ckpts = [args.ckpt]
        file_path = Path(args.ckpt) / '-fid_result.txt'
        plot_path = None
    else:
        raise FileNotFoundError("something wrong with ckpt path")
    logging.info(f"calculate the following {len(ckpts)} ckpt files: {[str(ckpt) for ckpt in ckpts]}")
        
    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()
    
    if args.inception is not None:
        logging.info("load inception from cache file")
        with open(args.inception, 'rb') as f:
            embeds = pickle.load(f)
            real_mean = embeds['mean']
            real_cov = embeds['cov']
    else:
        logging.info('calculating inception ...')
        real_mean, real_cov = extract_feature_from_real_images(config, args, inception, device)
        logging.info("save inception cache to inception_cache.pkl...")
        with open('inception_cache.pkl', 'wb') as f:
            pickle.dump(dict(mean=real_mean, cov=real_cov), f)
    logging.info("complete")
    
    

    names, fids = [], []
    for ckpt in ckpts:
        print(f"\r calculating fid of {str(ckpt)}", end="")
        names.append(int(ckpt.name[5:11])/1000)
        ckpt = torch.load(ckpt)
        
        # latent_dim, label_size, resolution
        g = Generator(config.MODEL.LATENT_SIZE, 0, config.RESOLUTION,
                      extra_channels=config.MODEL.EXTRA_CHANNEL).to(device)
        g.load_state_dict(ckpt['g_ema'])
        g = nn.DataParallel(g)
        g.eval()

        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = g.mean_latent(args.truncation_mean)
        else:
            mean_latent = None

        features = extract_feature_from_ckpt(
            g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
        ).numpy()
        print(f'extracted {features.shape[0]} features')

        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
        fids.append(fid)
        print('fid:', fid)
        
    with open(file_path, 'a') as f:
        for name, fid in zip(names, fids):
            f.write(f'{name}: {fid}\n')

    if plot_path:
        plt.plot(names, fids)
        plt.xlabel('k-iterations')
        plt.ylabel('fid')
        plt.savefig(plot_path)


if __name__ == '__main__':
    device = 'cuda'
    import sys
    parser = argparse.ArgumentParser()

    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument("--cfg", required=True, help="path to the configuration file")
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--inception', type=str, default=None)
    parser.add_argument('ckpt', metavar='CHECKPOINT', help='model ckpt or dir')
#     parser.add_argument('--extra_channels', type=int, default=3)

    args = parser.parse_args()
    update_config(config, args)
    args.ckpt = Path(args.ckpt)
    fid(config, args)
    

        