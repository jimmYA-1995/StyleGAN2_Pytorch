import pickle

import numpy as np
import torch

nv2my_G = {
'lod': '',
'dlatent_avg': '',
'G_synthesis/noise0': '',
'G_synthesis/noise1': '',
'G_synthesis/noise2': '',
'G_synthesis/noise3': '',
'G_synthesis/noise4': '',
'G_synthesis/noise5': '',
'G_synthesis/noise6': '',
'G_synthesis/noise7': '',
'G_synthesis/noise8': '',
'G_synthesis/noise9': '',
'G_synthesis/noise10': '',
'G_synthesis/noise11': '',
'G_synthesis/noise12': '',
'G_synthesis/4x4/Const/const': 'synthesis_network.input',
'G_synthesis/4x4/Conv/weight': 'synthesis_network.bottom_layer.conv.w',
'G_synthesis/4x4/Conv/mod_weight': 'synthesis_network.bottom_layer.conv.dense.w',
'G_synthesis/4x4/Conv/mod_bias': 'synthesis_network.bottom_layer.conv.dense.b',
'G_synthesis/4x4/Conv/noise_strength': 'synthesis_network.bottom_layer.noise.noise_strength',
'G_synthesis/4x4/Conv/bias': 'synthesis_network.bottom_layer.act.bias',
'G_synthesis/4x4/ToRGB/weight': 'synthesis_network.trgbs.0.conv.w',
'G_synthesis/4x4/ToRGB/mod_weight': 'synthesis_network.trgbs.0.conv.dense.w',
'G_synthesis/4x4/ToRGB/mod_bias': 'synthesis_network.trgbs.0.conv.dense.b',
'G_synthesis/4x4/ToRGB/bias': 'synthesis_network.trgbs.0.bias',
'G_synthesis/8x8/Conv0_up/weight': 'synthesis_network.convs.0.conv.w',
'G_synthesis/8x8/Conv0_up/mod_weight': 'synthesis_network.convs.0.conv.dense.w',
'G_synthesis/8x8/Conv0_up/mod_bias': 'synthesis_network.convs.0.conv.dense.b',
'G_synthesis/8x8/Conv0_up/noise_strength': 'synthesis_network.convs.0.noise.noise_strength',
'G_synthesis/8x8/Conv0_up/bias': 'synthesis_network.convs.0.act.bias',
'G_synthesis/8x8/Conv1/weight': 'synthesis_network.convs.1.conv.w',
'G_synthesis/8x8/Conv1/mod_weight': 'synthesis_network.convs.1.conv.dense.w',
'G_synthesis/8x8/Conv1/mod_bias': 'synthesis_network.convs.1.conv.dense.b',
'G_synthesis/8x8/Conv1/noise_strength': 'synthesis_network.convs.1.noise.noise_strength',
'G_synthesis/8x8/Conv1/bias': 'synthesis_network.convs.1.act.bias',
'G_synthesis/8x8/ToRGB/weight': 'synthesis_network.trgbs.1.conv.w',
'G_synthesis/8x8/ToRGB/mod_weight': 'synthesis_network.trgbs.1.conv.dense.w',
'G_synthesis/8x8/ToRGB/mod_bias': 'synthesis_network.trgbs.1.conv.dense.b',
'G_synthesis/8x8/ToRGB/bias': 'synthesis_network.trgbs.1.bias',
'G_synthesis/16x16/Conv0_up/weight': 'synthesis_network.convs.2.conv.w',
'G_synthesis/16x16/Conv0_up/mod_weight': 'synthesis_network.convs.2.conv.dense.w',
'G_synthesis/16x16/Conv0_up/mod_bias': 'synthesis_network.convs.2.conv.dense.b',
'G_synthesis/16x16/Conv0_up/noise_strength': 'synthesis_network.convs.2.noise.noise_strength',
'G_synthesis/16x16/Conv0_up/bias': 'synthesis_network.convs.2.act.bias',
'G_synthesis/16x16/Conv1/weight':  'synthesis_network.convs.3.conv.w',
'G_synthesis/16x16/Conv1/mod_weight':  'synthesis_network.convs.3.conv.dense.w',
'G_synthesis/16x16/Conv1/mod_bias':  'synthesis_network.convs.3.conv.dense.b',
'G_synthesis/16x16/Conv1/noise_strength':  'synthesis_network.convs.3.noise.noise_strength',
'G_synthesis/16x16/Conv1/bias':  'synthesis_network.convs.3.act.bias',
'G_synthesis/16x16/ToRGB/weight': 'synthesis_network.trgbs.2.conv.w',
'G_synthesis/16x16/ToRGB/mod_weight': 'synthesis_network.trgbs.2.conv.dense.w',
'G_synthesis/16x16/ToRGB/mod_bias': 'synthesis_network.trgbs.2.conv.dense.b',
'G_synthesis/16x16/ToRGB/bias': 'synthesis_network.trgbs.2.bias',
'G_synthesis/32x32/Conv0_up/weight':  'synthesis_network.convs.4.conv.w',
'G_synthesis/32x32/Conv0_up/mod_weight':  'synthesis_network.convs.4.conv.dense.w',
'G_synthesis/32x32/Conv0_up/mod_bias':  'synthesis_network.convs.4.conv.dense.b',
'G_synthesis/32x32/Conv0_up/noise_strength':  'synthesis_network.convs.4.noise.noise_strength',
'G_synthesis/32x32/Conv0_up/bias':  'synthesis_network.convs.4.act.bias',
'G_synthesis/32x32/Conv1/weight':  'synthesis_network.convs.5.conv.w',
'G_synthesis/32x32/Conv1/mod_weight':  'synthesis_network.convs.5.conv.dense.w',
'G_synthesis/32x32/Conv1/mod_bias':  'synthesis_network.convs.5.conv.dense.b',
'G_synthesis/32x32/Conv1/noise_strength':  'synthesis_network.convs.5.noise.noise_strength',
'G_synthesis/32x32/Conv1/bias':  'synthesis_network.convs.5.act.bias',
'G_synthesis/32x32/ToRGB/weight': 'synthesis_network.trgbs.3.conv.w',
'G_synthesis/32x32/ToRGB/mod_weight': 'synthesis_network.trgbs.3.conv.dense.w',
'G_synthesis/32x32/ToRGB/mod_bias': 'synthesis_network.trgbs.3.conv.dense.b',
'G_synthesis/32x32/ToRGB/bias': 'synthesis_network.trgbs.3.bias',
'G_synthesis/64x64/Conv0_up/weight':  'synthesis_network.convs.6.conv.w',
'G_synthesis/64x64/Conv0_up/mod_weight':  'synthesis_network.convs.6.conv.dense.w',
'G_synthesis/64x64/Conv0_up/mod_bias':  'synthesis_network.convs.6.conv.dense.b',
'G_synthesis/64x64/Conv0_up/noise_strength':  'synthesis_network.convs.6.noise.noise_strength',
'G_synthesis/64x64/Conv0_up/bias':  'synthesis_network.convs.6.act.bias',
'G_synthesis/64x64/Conv1/weight':  'synthesis_network.convs.7.conv.w',
'G_synthesis/64x64/Conv1/mod_weight':  'synthesis_network.convs.7.conv.dense.w',
'G_synthesis/64x64/Conv1/mod_bias':  'synthesis_network.convs.7.conv.dense.b',
'G_synthesis/64x64/Conv1/noise_strength':  'synthesis_network.convs.7.noise.noise_strength',
'G_synthesis/64x64/Conv1/bias':  'synthesis_network.convs.7.act.bias',
'G_synthesis/64x64/ToRGB/weight': 'synthesis_network.trgbs.4.conv.w',
'G_synthesis/64x64/ToRGB/mod_weight': 'synthesis_network.trgbs.4.conv.dense.w',
'G_synthesis/64x64/ToRGB/mod_bias': 'synthesis_network.trgbs.4.conv.dense.b',
'G_synthesis/64x64/ToRGB/bias': 'synthesis_network.trgbs.4.bias',
'G_synthesis/128x128/Conv0_up/weight':  'synthesis_network.convs.8.conv.w',
'G_synthesis/128x128/Conv0_up/mod_weight':  'synthesis_network.convs.8.conv.dense.w',
'G_synthesis/128x128/Conv0_up/mod_bias':  'synthesis_network.convs.8.conv.dense.b',
'G_synthesis/128x128/Conv0_up/noise_strength':  'synthesis_network.convs.8.noise.noise_strength',
'G_synthesis/128x128/Conv0_up/bias':  'synthesis_network.convs.8.act.bias',
'G_synthesis/128x128/Conv1/weight':  'synthesis_network.convs.9.conv.w',
'G_synthesis/128x128/Conv1/mod_weight':  'synthesis_network.convs.9.conv.dense.w',
'G_synthesis/128x128/Conv1/mod_bias':  'synthesis_network.convs.9.conv.dense.b',
'G_synthesis/128x128/Conv1/noise_strength':  'synthesis_network.convs.9.noise.noise_strength',
'G_synthesis/128x128/Conv1/bias':  'synthesis_network.convs.9.act.bias',
'G_synthesis/128x128/ToRGB/weight': 'synthesis_network.trgbs.5.conv.w',
'G_synthesis/128x128/ToRGB/mod_weight': 'synthesis_network.trgbs.5.conv.dense.w',
'G_synthesis/128x128/ToRGB/mod_bias': 'synthesis_network.trgbs.5.conv.dense.b',
'G_synthesis/128x128/ToRGB/bias': 'synthesis_network.trgbs.5.bias',
'G_synthesis/256x256/Conv0_up/weight':  'synthesis_network.convs.10.conv.w',
'G_synthesis/256x256/Conv0_up/mod_weight':  'synthesis_network.convs.10.conv.dense.w',
'G_synthesis/256x256/Conv0_up/mod_bias':  'synthesis_network.convs.10.conv.dense.b',
'G_synthesis/256x256/Conv0_up/noise_strength':  'synthesis_network.convs.10.noise.noise_strength',
'G_synthesis/256x256/Conv0_up/bias':  'synthesis_network.convs.10.act.bias',
'G_synthesis/256x256/Conv1/weight':  'synthesis_network.convs.11.conv.w',
'G_synthesis/256x256/Conv1/mod_weight':  'synthesis_network.convs.11.conv.dense.w',
'G_synthesis/256x256/Conv1/mod_bias':  'synthesis_network.convs.11.conv.dense.b',
'G_synthesis/256x256/Conv1/noise_strength':  'synthesis_network.convs.11.noise.noise_strength',
'G_synthesis/256x256/Conv1/bias':  'synthesis_network.convs.11.act.bias',
'G_synthesis/256x256/ToRGB/weight': 'synthesis_network.trgbs.6.conv.w',
'G_synthesis/256x256/ToRGB/mod_weight': 'synthesis_network.trgbs.6.conv.dense.w',
'G_synthesis/256x256/ToRGB/mod_bias': 'synthesis_network.trgbs.6.conv.dense.b',
'G_synthesis/256x256/ToRGB/bias': 'synthesis_network.trgbs.6.bias',
'G_mapping/Dense0/weight': 'mapping_network.fc.0.w',
'G_mapping/Dense0/bias': 'mapping_network.fc.0.b',
'G_mapping/Dense1/weight': 'mapping_network.fc.1.w',
'G_mapping/Dense1/bias': 'mapping_network.fc.1.b',
'G_mapping/Dense2/weight': 'mapping_network.fc.2.w',
'G_mapping/Dense2/bias': 'mapping_network.fc.2.b',
'G_mapping/Dense3/weight': 'mapping_network.fc.3.w',
'G_mapping/Dense3/bias': 'mapping_network.fc.3.b',
'G_mapping/Dense4/weight': 'mapping_network.fc.4.w',
'G_mapping/Dense4/bias': 'mapping_network.fc.4.b',
'G_mapping/Dense5/weight': 'mapping_network.fc.5.w',
'G_mapping/Dense5/bias': 'mapping_network.fc.5.b',
'G_mapping/Dense6/weight': 'mapping_network.fc.6.w',
'G_mapping/Dense6/bias': 'mapping_network.fc.6.b',
'G_mapping/Dense7/weight': 'mapping_network.fc.7.w',
'G_mapping/Dense7/bias': 'mapping_network.fc.7.b',
}

nv2my_D = {
'256x256/FromRGB/weight': 'frgb.conv.w',
'256x256/FromRGB/bias': 'frgb.conv.bias_act.bias',
'256x256/Conv0/weight': 'blocks.0.conv.conv.w',
'256x256/Conv0/bias': 'blocks.0.conv.act.bias',
'256x256/Conv1_down/weight': 'blocks.0.conv_down.conv.w',
'256x256/Conv1_down/bias': 'blocks.0.conv_down.act.bias',
'256x256/Skip/weight': 'blocks.0.skip.w',
'128x128/Conv0/weight': 'blocks.1.conv.conv.w',
'128x128/Conv0/bias': 'blocks.1.conv.act.bias',
'128x128/Conv1_down/weight': 'blocks.1.conv_down.conv.w',
'128x128/Conv1_down/bias': 'blocks.1.conv_down.act.bias',
'128x128/Skip/weight': 'blocks.1.skip.w',
'64x64/Conv0/weight': 'blocks.2.conv.conv.w',
'64x64/Conv0/bias': 'blocks.2.conv.act.bias',
'64x64/Conv1_down/weight': 'blocks.2.conv_down.conv.w',
'64x64/Conv1_down/bias': 'blocks.2.conv_down.act.bias',
'64x64/Skip/weight': 'blocks.2.skip.w',
'32x32/Conv0/weight': 'blocks.3.conv.conv.w',
'32x32/Conv0/bias': 'blocks.3.conv.act.bias',
'32x32/Conv1_down/weight': 'blocks.3.conv_down.conv.w',
'32x32/Conv1_down/bias': 'blocks.3.conv_down.act.bias',
'32x32/Skip/weight': 'blocks.3.skip.w',
'16x16/Conv0/weight': 'blocks.4.conv.conv.w',
'16x16/Conv0/bias': 'blocks.4.conv.act.bias',
'16x16/Conv1_down/weight': 'blocks.4.conv_down.conv.w',
'16x16/Conv1_down/bias': 'blocks.4.conv_down.act.bias',
'16x16/Skip/weight': 'blocks.4.skip.w',
'8x8/Conv0/weight': 'blocks.5.conv.conv.w',
'8x8/Conv0/bias': 'blocks.5.conv.act.bias',
'8x8/Conv1_down/weight': 'blocks.5.conv_down.conv.w',
'8x8/Conv1_down/bias': 'blocks.5.conv_down.act.bias',
'8x8/Skip/weight': 'blocks.5.skip.w',
'4x4/Conv/weight': 'conv_out.conv.w',
'4x4/Conv/bias': 'conv_out.act.bias',
'4x4/Dense0/weight': 'dense_out.w',
'4x4/Dense0/bias': 'dense_out.b',
'Output/weight': 'label_out.w',
'Output/bias': 'label_out.b',
}

def load_weights_from_nv(g, d, g_ema, path):
    
    with open(path, 'rb') as f:
        w = pickle.load(f)
        
    g_dict = {k: v for k,v in g.named_parameters()}
    g_ema_dict = {k: v for k,v in g_ema.named_parameters()}
    
    for nv, my in nv2my_G.items():
        if my == '':
            continue

        nv_w = w['G'][nv]
        nv_w1 = w['G_ema'][nv]
        if 'conv.w' in my:
            nv_w = np.transpose(nv_w, (3,2,0,1))
            nv_w1 = np.transpose(nv_w1, (3,2,0,1))
        if 'conv.dense.w' in my:
            nv_w = np.transpose(nv_w)
            nv_w1 = np.transpose(nv_w1)
        if 'noise_strength' in my:
            nv_w = nv_w[..., None]
            nv_w1 = nv_w1[..., None]
        if 'trgbs' in my and 'bias' in my:
            nv_w = np.expand_dims(nv_w, [0,2,3])
            nv_w1 = np.expand_dims(nv_w1, [0,2,3])

        with torch.no_grad():
            g_dict[my].copy_(torch.from_numpy(nv_w))
            g_ema_dict[my].copy_(torch.from_numpy(nv_w))

    d_dict = {k: v for k,v in d.named_parameters()}
    
    for nv, my in nv2my_D.items():
        nv_w = w['D'][nv]
        if 'conv.w' in my or 'skip.w' in my:
            nv_w = np.transpose(nv_w, (3,2,0,1))
        if my in ['dense_out.w', 'label_out.w']:
            nv_w = np.transpose(nv_w)

        with torch.no_grad():
            d_dict[my].copy_(torch.from_numpy(nv_w))
            

            
def load_partial_weights(g, d, g_ema, ckpt, logger=None):
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
        if logger:
            logger.info("Transfer learning. Set start iteration to 0")
        return 0
    except RuntimeError:
        if logger:
            logger.error(" ***** fail to load partial weights to models ***** ")
        raise RuntimeError("fail to load partial weights to models. Please check your checkpoint")