import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm

from calc_inception import load_patched_inception_v3



def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class FIDTracker():
    def __init__(self, config, dataloader, output_dir, logger, use_tqdm=False):
        inception_path = config.INCEPTION_CACHE
        self.device = 'cuda'
        self.config = config
        self.logger = logger
        self.output_path = Path(output_dir)
        self.k_iters = []
        self.fids = []
        self.n_batch = config.N_SAMPLE // config.BATCH_SIZE
        self.resid = config.N_SAMPLE % config.BATCH_SIZE
        self.idx_iterator = range(self.n_batch + 1)
        if use_tqdm:
            self.idx_iterator = tqdm(self.idx_iterator)
        
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
                self.extract_feature_from_real_images(dataloader)
            self.logger.info(f"save inception cache in {self.output_path}")
            with open(self.output_path / 'inception_cache.pkl', 'wb') as f:
                pickle.dump(dict(mean=self.real_mean, cov=self.real_cov), f)

    def calc_fid(self, generator, k_iter, save=False, eps=1e-6):
        self.logger.info(f'get fid on {k_iter * 1000} iterations')
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
        self.logger.info(f'FID in {str(1000 * k_iter).zfill(6)} iterations: {fid}')
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
        imgs = next(loader)
        if isinstance(imgs, (tuple, list)):
            imgs = imgs[0]
            
        bs = imgs.shape[0]
        resid = self.config.N_SAMPLE % bs
        n_batch = self.config.N_SAMPLE // bs
        
        features = []
        
        for i in tqdm(range(n_batch + 1)):
            batch_size = resid if i==n_batch else bs
            if batch_size == 0:
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
            if batch==0:
                continue
            
            latent = torch.randn(batch, 512, device=self.device)
            img, _ = generator([latent])
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

