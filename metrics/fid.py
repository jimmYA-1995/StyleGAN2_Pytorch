import time
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from scipy import linalg
from tqdm import tqdm

from dataset import get_dataloader_for_each_class
from calc_inception import load_patched_inception_v3

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def load_condition_sample(sample_dir, batch_size):
    IMG_EXTS = ['jpg', 'png', 'jpeg']
    samples = []
    for ext in IMG_EXTS:
        samples.extend(list(Path(sample_dir).glob(f'*.{ext}')))
    assert len(samples) >= batch_size, f"Need more samples. {len(samples)} < {batch_size}"
    cond_samples = []
                
    trf = [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [5],
                                     inplace=True),
            ]
    transform = transforms.Compose(trf)
    
    for i, p in enumerate():
        if batch_size != -1 and i == batch_size:
            break
        cond_samples.append(transform(io.imread(p)[..., None])[None, ...])
    return torch.cat(cond_samples, dim=0)
            
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
        self.real_mean = None # ndarray(n_class, 2048) float32
        self.real_cov = None # ndarray(n_class, 2048, 2048) float64
        self.fids = []
        self.conditional = True if config.N_CLASSES > 1 else False 
        self.num_classes = config.N_CLASSES
        self.cond_samples = None
        self.n_batch = fid_config.N_SAMPLE // fid_config.BATCH_SIZE
        self.resid = fid_config.N_SAMPLE % fid_config.BATCH_SIZE
        self.idx_iterator = range(self.n_batch + 1)
        self.use_tqdm = use_tqdm
        self.model_bs = config.N_SAMPLE
        if fid_config.SAMPLE_DIR:
            self.cond_samples = load_condition_sample(fid_config.SAMPLE_DIR, self.model_bs)
            self.cond_samples = self.cond_samples.to(self.device)
            self.logger.info(f"using smaple directory: {fid_config.SAMPLE_DIR}. \
                                Get {self.cond_samples.shape[0]} conditional sample")
            self.model_bs = self.cond_samples.shape[0]

        
        # get inception V3 model
        start = time.time()
        self.logger.info("load inception model...")
        self.inceptionV3 = torch.nn.DataParallel(load_patched_inception_v3()).to(self.device)
        self.inceptionV3.eval()
        self.logger.info("load inception model complete ({:.2f})".format(time.time() - start))
 
        # get features for real images
        if inception_path:
            self.logger.info("load inception from cache file")
            with open(inception_path, 'rb') as f:
                embeds = pickle.load(f)
                self.real_mean = embeds['mean']
                self.real_cov = embeds['cov']
                self.idx_to_class = embeds['idx_to_class']
        else:
            self.real_mean, self.real_cov, self.idx_to_class = self.extract_feature_from_real_images(config)
            self.logger.info(f"save inception cache in {self.output_path}")
            with open(self.output_path / 'inception_cache.pkl', 'wb') as f:
                pickle.dump(dict(mean=self.real_mean, cov=self.real_cov, idx_to_class=self.idx_to_class), f)

    def calc_fid(self, generator, k_iter, save=False, eps=1e-6):
        self.logger.info(f'get fid on {k_iter * 1000} iterations')
        start = time.time()
        sample_features = self.extract_feature_from_model(generator)
        
        fids = []
        for i, sample_feature in enumerate(sample_features):
            sample_mean = np.mean(sample_feature, 0)
            sample_cov = np.cov(sample_feature, rowvar=False)

            cov_sqrt, _ = linalg.sqrtm(sample_cov @ self.real_cov[i], disp=False)

            if not np.isfinite(cov_sqrt).all():
                self.logger.warning('product of cov matrices is singular')
                offset = np.eye(sample_cov.shape[0]) * eps
                cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (self.real_cov[i] + offset))

            if np.iscomplexobj(cov_sqrt):
                if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                    m = np.max(np.abs(cov_sqrt.imag))

                    raise ValueError(f'Imaginary component {m}')

                cov_sqrt = cov_sqrt.real

            mean_diff = sample_mean - self.real_mean[i]
            mean_norm = mean_diff @ mean_diff

            trace = np.trace(sample_cov) + np.trace(self.real_cov[i]) - 2 * np.trace(cov_sqrt)
            fids.append(mean_norm + trace)
            
        finish = time.time()
        self.logger.info(f'FID in {str(1000 * k_iter).zfill(6)} \
             iterations: "{fids}". [costs {round(finish - start, 2)} sec(s)]')
        self.k_iters.append(k_iter)
        self.fids.append(fids)
    
        if save:
            with open(self.output_path / 'fid.txt', 'a+') as f:
                f.write(f'{k_iter}: {fids}\n')
        
        return fids

    @torch.no_grad()
    def extract_feature_from_real_images(self, config):
        dataloaders, idx_to_class = get_dataloader_for_each_class(config, self.config.BATCH_SIZE)
        assert self.num_classes == len(idx_to_class), "N_CLASSES in user config not equal to #class in dataset"
        start = time.time()
        
        real_mean_list = []
        real_cov_list = []
        
        for i, dataloader in enumerate(dataloaders):
            self.logger.info(f'extract features from real "{idx_to_class[i]}" images...')
            features = []
            loader = sample_data(dataloader)
        
            if self.use_tqdm:
                idx_iterator = tqdm(self.idx_iterator)
            for i in idx_iterator:
                batch = self.resid if i==self.n_batch else self.config.BATCH_SIZE
                if batch == 0:
                    continue

                imgs, _ = next(loader)
                imgs = imgs[:batch, :3, :, :].to(self.device)
                feature = self.inceptionV3(imgs)[0].view(imgs.shape[0], -1)
                features.append(feature.to('cpu'))

            features = torch.cat(features, 0).numpy()
            real_mean = np.mean(features, 0)
            real_cov = np.cov(features, rowvar=False)
            real_mean_list.append(real_mean)
            real_cov_list.append(real_cov)

            self.logger.info(f"complete({round(time.time() - start, 2)} secs). \
                               total extracted features: {features.shape[0]}")
            
        return real_mean_list, real_cov_list, idx_to_class

    @torch.no_grad()
    def extract_feature_from_model(self, generator):
        n_batch = self.config.N_SAMPLE // self.model_bs
        resid = self.config.N_SAMPLE % self.model_bs
        features_list = []
        
        for class_idx in range(self.num_classes):
            features = []
            idx_iterator = range(n_batch + 1)
            if self.use_tqdm:
                idx_iterator = tqdm(idx_iterator)

            for i in idx_iterator:
                batch = resid if i==n_batch else self.model_bs
                if batch == 0:
                    continue

                latent = torch.randn(batch, 512, device=self.device)
                fake_label = torch.LongTensor([class_idx]*batch).to(self.device)
                
                cond_samples=None
                if self.cond_samples is not None:
                    cond_samples = self.cond_samples[:batch]

                img, _ = generator([latent], labels_in=fake_label, sk=cond_samples)
                feature = self.inceptionV3(img[:, :3, :, :])[0].view(img.shape[0], -1)
                features.append(feature.to('cpu'))

            features_list.append(torch.cat(features, 0).numpy())
        return features_list
    
    def plot_fid(self,):
        self.logger.info(f"save FID figure in {str(self.output_path / 'fid.png')}")
        
        self.fids = np.array(self.fids).T
        plt.legend(tuple([self.idx_to_class[idx] for idx in range(self.num_classes)]),  loc='upper right')
        plt.xlabel('k iterations')
        plt.ylabel('FID')
        for fids in self.fids:
            plt.plot(self.k_iters, fids)
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

    use_cond_sample = True if config.EVAL.FID.SAMPLE_DIR else False
    fid_tracker = FIDTracker(config, args.out_dir)

    if not args.ckpt:
        print("checkpoint(s) not found. Only get features of real images.\n return...")
        exit(0)

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
                      extra_channels=config.MODEL.EXTRA_CHANNEL, use_sk=use_cond_sample).to(device)
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
