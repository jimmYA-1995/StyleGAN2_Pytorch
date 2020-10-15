import time
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from config import config, update_config
from dataset import get_dataloader_for_each_class


class AccuracyTracker():
    def __init__(self, config, out_dir, use_tqdm=False):
        self.device = 'cuda'
        self.out_dir = out_dir
        self.num_classes = config.N_CLASSES
        self.batch_size = config.EVAL.ACC.BATCH_SIZE
        self.n_samples = config.EVAL.ACC.N_SAMPLE
        self.loaders, self.idx_to_class = get_dataloader_for_each_class(config, self.batch_size, is_validation=True)
        self.logger = logging.getLogger()
        self.output_path = Path(out_dir)
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        self.k_iters = []
        self.accs = []

    @torch.no_grad()
    def calc_accuracy(self, discriminator, k_iter, save=False):
        self.logger.info(f'get accuracy on {k_iter * 1000} iterations')
        start = time.time()
        accs = []
        
        for loader in self.loaders:
            total = 0
            correct = 0
            for i, (real_img, labels) in enumerate(loader):
                real_img = real_img.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)
                    
                _, real_pred_C = discriminator(real_img)
                pred = torch.argmax(nn.Softmax(dim=1)(real_pred_C), dim=1)
                if self.n_samples != -1 and i*self.batch_size > self.n_samples:
                    reside = self.n_samples % self.batch_size
                else:
                    reside = self.batch_size

                correct += (pred[:reside]==labels[:reside]).cpu().numpy().astype(int).sum()
                total += reside
            accs.append(correct/total)
                    
        finish = time.time()
        self.logger.info(f'Accuracy in {str(1000 * k_iter).zfill(6)} \
             iterations: "{accs}". [costs {round(finish - start, 2)} sec(s)]')
        self.k_iters.append(k_iter)
        self.accs.append(accs)
        if save:
            with open(self.output_path / 'D_accuracy.txt', 'a+') as f:
                f.write(f'{k_iter}: {accs}\n')
        
        return 
        
    def plot_result(self):
        acc_path = str(self.output_path / 'D_accuracy.png')
        self.logger.info(f"save Accuracy figure in {acc_path}")

        self.accs = np.array(self.accs).T
        plt.xlabel('k iterations')
        plt.ylabel('Accuracy')
        for acc in self.accs:
            plt.plot(self.k_iters, acc)
        plt.legend([self.idx_to_class[idx] for idx in range(self.num_classes)], loc='upper right')
        plt.savefig(acc_path)

class Args:
    cfg='experiments/itri-ac.yml'

if __name__ == "__main__":
    from models import Discriminator
    logging.basicConfig(level='INFO')
    device = 'cuda'
    update_config(config, Args())
    tracker = AccuracyTracker(config, './')
    
    assert config.N_CLASSES >= 1
    label_size = 0 if config.N_CLASSES == 1 else config.N_CLASSES
    d = Discriminator(label_size, config.RESOLUTION, extra_channels=config.MODEL.EXTRA_CHANNEL).to(device)
    ckpt = torch.load('results/00025-1gpu-itri/checkpoints/ckpt-147500.pt')
    d.load_state_dict(ckpt['d'])
    d = nn.DataParallel(d)
    d.eval()
    
    tracker.calc_accuracy(d, 1, save=True)
    tracker.plot_result()