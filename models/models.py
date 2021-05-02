import time
import random
import logging
import importlib
import numpy as np
import torch
import torch.nn as nn

from .components import FromRGB, DBlock, minibatch_stddev_layer, Layer, Dense_layer

logger = logging.getLogger()

class Generator(nn.Module):
    def __init__(
            self,
            latent_size, label_size, resolution, embedding_size=0,
            dlatents_size=512, is_training=None, return_dlatents=False,
            truncation_psi=0.5, truncation_cut_off=None,
            truncation_psi_val=None, truncation_cut_off_val=None,
            dlatent_avg_beta=0.995,
            mapping_network='G_mapping',
            synthesis_netowrk='G_synthesis_stylegan2',
            randomize_noise=True,
            extra_channels=0,
            **kwargs):
        super(Generator, self).__init__()
        
        if label_size > 0:
            assert embedding_size > 0 
        # assert is_training is not None
        # if is_training:
        #     truncation_psi = truncation_cut_off = None
        # else: # ignore some condition.
        #     truncation_psi = truncation_psi_val
        #     truncation_cut_off = truncation_cut_off_val
        #     dlatent_avg_beta = None
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        assert resolution == 2**self.resolution_log2 and resolution >= 4
        
        self.return_dlatents = return_dlatents
        
        self.num_channels = 3 + extra_channels
            
        # Define arch. of components
        mapping_class = getattr(
            importlib.import_module('.components', 'models'), mapping_network
        )
        synthesis_class = getattr(
            importlib.import_module('.components', 'models'), synthesis_netowrk
        )
        self.mapping_network = mapping_class(
            latent_size, label_size, embedding_size, dlatents_size
        )
        self.synthesis_network = synthesis_class(
            self.num_layers, self.resolution_log2,
            num_channels=self.num_channels,
            randomize_noise=randomize_noise,
            dlatents_size=dlatents_size, architecture='skip'
        )

    def forward(self, latents_in, labels_in=None, style_in=None, content_in=None, return_latents=None):
        # style mixing
        if len(latents_in) == 2:
            idx = random.randint(1, self.num_layers - 1)
            inject_index = [idx, self.num_layers - idx] ## name confusing
            dlatents = [self.mapping_network(l, labels_in, dlatent_broadcast=i)
                        for l, i in zip(latents_in, inject_index)]
            dlatents = torch.cat(dlatents, dim=1)
        else:
            dlatents = self.mapping_network(latents_in[0], labels_in,
                                            dlatent_broadcast=self.num_layers)
            
        images_out = self.synthesis_network(dlatents, style_in=style_in, content_in=content_in)
        if return_latents:
            return images_out, dlatents
        return images_out, None

    def get_latent(self, latents_in, labels_in=None, return_latents=None):
        # style mixing
        if len(latents_in) == 2:
            idx = random.randint(1, self.num_layers - 1)
            inject_index = [idx, self.num_layers - idx] ## name confusing
            dlatents = [self.mapping_network(l, labels_in, dlatent_broadcast=i)
                        for l, i in zip(latents_in, inject_index)]
            dlatents = torch.cat(dlatents, dim=1)
        else:
            dlatents = self.mapping_network(latents_in[0], labels_in,
                                            dlatent_broadcast=self.num_layers)
        return dlatents
    
class Discriminator(nn.Module):
    def __init__(
            self, label_size, resolution, extra_channels=3,
            fmap_base = 16<<10, fmap_decay=1.0, fmap_min=1, fmap_max=512,          
            mbstd_group_size=4, mbstd_num_features=1,             
            resample_kernel=[1,3,3,1], architecture='resnet',
            **kwargs):
        super(Discriminator, self).__init__()
        assert architecture in ['skip', 'resnet'], "unsupported D architecture."
        self.img_channels = 3 + extra_channels
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features
        self.resolution_log2 = int(np.log2(resolution))
        self.arch = architecture
        def nf(stage):
            scaled = int(fmap_base / (2.0 ** (stage * fmap_decay)))
            return np.clip(scaled, fmap_min, fmap_max)
        
        if self.arch == 'resnet':
            self.frgb = FromRGB(self.img_channels, nf(self.resolution_log2-1))

        self.blocks = nn.ModuleList()
        
        for res in range(self.resolution_log2, 2, -1):
            self.blocks.append(DBlock(nf(res-1), nf(res-2), self.arch))
        
        # output layer
        self.conv_out = Layer(nf(1)+1, nf(1), kernel=3, use_bias=False)
        self.dense_out = Dense_layer(512 * 4 * 4, nf(0))
        self.label_out = Dense_layer(nf(0), max(label_size, 1))

    def forward(self, images_in, labels_in=None):
        assert images_in.shape[1] == self.img_channels, \
               f"(D) channel unmatched. {images_in.shape[1]} v.s. {self.img_channels}"
        x = None
        skip = None
        if self.arch == 'resnet':
            x = self.frgb(images_in)
        else:
            skip = images_in
            
        for block in self.blocks:
            x, skip = block(x, skip)
            
        # TODO: FRGB if skip
        x = minibatch_stddev_layer(
            x, self.mbstd_group_size, self.mbstd_num_features
        )
        x = self.conv_out(x)
        x = x.view(x.shape[0], -1)
        x = self.dense_out(x)
        out = self.label_out(x)
        if labels_in is not None and labels_in.shape[1] > 0:
            out = torch.mean(out * labels_in, dim=1, keepdims=True)
            
        return out
