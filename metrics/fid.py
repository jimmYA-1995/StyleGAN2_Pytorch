import time
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm

from dataset import get_dataloader
from calc_inception import load_patched_inception_v3

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class FIDTracker():
    def __init__(self, config, output_dir, use_tqdm=False):

        fid_config = config.EVAL.FID
        inception_path = fid_config.INCEPTION_CACHE
        self.device = 'cuda'
        self.config = fid_config
        self.logger = logging.getLogger()
        self.output_path = Path(output_dir)
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        self.k_iters = []
        self.fids = []
        self.sample_sk = None
        self.n_batch = fid_config.N_SAMPLE // fid_config.BATCH_SIZE
        self.resid = fid_config.N_SAMPLE % fid_config.BATCH_SIZE
        self.idx_iterator = range(self.n_batch + 1)
        if use_tqdm:
            self.idx_iterator = tqdm(self.idx_iterator)
        
        if fid_config.BATCH_SIZE.SAMPLE_DIR:
            import skimage.io as io
            sample_sk = []
                
            trf = [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [5],
                                             inplace=True),
                    ]
            transform = transforms.Compose(trf)
            
            for p in Path(fid_config.SAMPLE_DIR).glob('*.jpg'):
                sample_sk.append(transform(io.imread(p)[..., None])[None, ...])
            self.sample_sk = torch.cat(sample_sk, dim=0).to('cuda')
            self.logger.info(f"using smaple directory: {fid_config.SAMPLE_DIR}. \
                                Get {self.sample_sk.shape[0]} skeleton sample")
        
        # get inception V3 model
        self.inceptionV3 = torch.nn.DataParallel(load_patched_inception_v3()).to(self.device)
        self.inceptionV3.eval()
 
        # get features for real images
        if inception_path:
            self.logger.info("load inception from cache file")
            with open(inception_path, 'rb') as f:
                embeds = pickle.load(f)
                self.real_mean = embeds['mean']
                self.real_cov = embeds['cov']
        else:
            self.real_mean, self.real_cov = \
                self.extract_feature_from_real_images(get_dataloader(config, fid_config.BATCH_SIZE))
            self.logger.info(f"save inception cache in {self.output_path}")
            with open(self.output_path / 'inception_cache.pkl', 'wb') as f:
                pickle.dump(dict(mean=self.real_mean, cov=self.real_cov), f)

    def calc_fid(self, generator, k_iter, save=False, eps=1e-6):
        self.logger.info(f'get fid on {k_iter * 1000} iterations')
        start = time.time()
        sample_features = self.extract_feature_from_model(generator)
        sample_mean = np.mean(sample_features, 0)
        sample_cov = np.cov(sample_features, rowvar=False)
        
        cov_sqrt, _ = linalg.sqrtm(sample_cov @ self.real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            self.logger.warning('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (self.real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - self.real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(self.real_cov) - 2 * np.trace(cov_sqrt)
        fid = mean_norm + trace
        finish = time.time()
        self.logger.info(f'FID in {str(1000 * k_iter).zfill(6)} \
             iterations: {fid}. [costs {round(finish - start, 2)} sec(s)]')
        self.k_iters.append(k_iter)
        self.fids.append(fid)
        
        if save:
            with open(self.output_path / 'fid.txt', 'a+') as f:
                f.write(f'{k_iter}: {fid}\n')
        
        return fid

    @torch.no_grad()
    def extract_feature_from_real_images(self, dataloder):
        self.logger.info('extract features from real images...')
        loader = sample_data(dataloder)
        features = []
        
        for i in self.idx_iterator:
            batch = self.resid if i==self.n_batch else self.config.BATCH_SIZE
            if batch == 0:
                continue

            imgs = next(loader)
            if isinstance(imgs, (tuple, list)):
                imgs = imgs[0]
            imgs = imgs[:batch_size, :3, :, :].to(self.device)
            feature = self.inceptionV3(imgs)[0].view(batch_size, -1)
            features.append(feature.to('cpu'))

        features = torch.cat(features, 0).numpy()
        self.logger.info(f"complete. total extracted features: {features.shape[0]}")

        real_mean = np.mean(features, 0)
        real_cov = np.cov(features, rowvar=False)
        return real_mean, real_cov

    @torch.no_grad()
    def extract_feature_from_model(self, generator):
        
        features = []
        for i in self.idx_iterator:
            batch = self.resid if i==self.n_batch else self.config.BATCH_SIZE
            if batch == 0:
                continue
            
            latent = torch.randn(batch, 512, device=self.device)
            if self.sample_sk is not None:
                sk = self.sample_sk[:batch]
            img, _ = generator([latent], sk=sk)
            feature = self.inceptionV3(img[:, :3, :, :])[0].view(img.shape[0], -1)
            features.append(feature.to('cpu'))

        features = torch.cat(features, 0)
        return features.numpy()
    
    def plot_fid(self,):
        self.logger.info(f"save FID figure in {str(self.output_path / 'fid.png')}")
        plt.plot(self.k_iters, self.fids)
        plt.xlabel('k iterations')
        plt.ylabel('FID')
        plt.savefig(self.output_path / 'fid.png')


if __name__ == '__main__':
    
    import sys
    from models import Generator
    from config import config, update_config
    from run_training import get_dataloader
    device = 'cuda'
    parser = argparse.ArgumentParser()

    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument("--cfg", required=True, help="path to the configuration file")
    parser.add_argument('--out_dir', type=str, default='/tmp/fid_result')
    parser.add_argument('--ckpt', type=str, default="", metavar='CHECKPOINT', help='model ckpt or dir')
    parser.add_argument("--debug", action='store_true', default=False, help="whether to use debug mode")
                      
    args = parser.parse_args()
    
    loglevel = "DEBUG" if args.debug else "INFO"
    head = "%(levelname)-8s - %(asctime)-15s - %(message)s (%(filename)s:%(lineno)d)"
    logging.basicConfig(level=loglevel,
                        format=head)
    logger = logging.getLogger()
    update_config(config, args)

    use_sk = True if config.EVAL.FID.SAMPLE_DIR else False
    fid_tracker = FIDTracker(config, args.out_dir)

    if not args.ckpt:
        print("checkpoint(s) not found. Only get features of real images.\n return...")
        return

    args.ckpt = Path(args.ckpt)
    ckpts = sorted(list(args.ckpt.glob('*.pt'))) \
            if args.ckpt.is_dir() else [args.ckpt]

    logger.info(f"Get FID of the following {len(ckpts)} ckpt files: {[str(ckpt) for ckpt in ckpts]}")
    
    for ckpt in ckpts:
        logging.info(f"calculating fid of {str(ckpt)}")
        ckpt_name = str(ckpt.name)[5:11]
        k_iter = int(ckpt_name)/1000
        ckpt = torch.load(ckpt)
        
        # latent_dim, label_size, resolution
        g = Generator(config.MODEL.LATENT_SIZE, 0, config.RESOLUTION,
                      extra_channels=config.MODEL.EXTRA_CHANNEL, use_sk=use_sk).to(device)
        g.load_state_dict(ckpt['g_ema'])
        g = nn.DataParallel(g)
        g.eval()

        # if args.truncation < 1:
        #     with torch.no_grad():
        #         mean_latent = g.mean_latent(args.truncation_mean)
        # else:
        #     mean_latent = None
        
        fid_tracker.calc_fid(g, k_iter, save=True)
    
    fid_tracker.plot_fid()