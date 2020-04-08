import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from op import upfirdn2d, FusedLeakyReLU, fused_leaky_relu

activation_funcs = {
    'linear':   lambda x, **_:        x,                          
    'relu':     lambda x, **_:        tf.nn.relu(x),              
    'lrelu':    lambda x, alpha, **_: tf.nn.leaky_relu(x, alpha), 
    'tanh':     lambda x, **_:        tf.nn.tanh(x),              
    'sigmoid':  lambda x, **_:        tf.nn.sigmoid(x),          
}

# from: https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    
    return k

    
def get_weight(shape, gain=1, use_wscale=True, lrmul=1):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul
        
    return Parameter(torch.randn(shape)), runtime_coef    
    

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out
    

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out

class Dense_layer(nn.Module):
    def __init__(self, in_dim, out_dim,
                 gain=1, use_wscale=True, lrmul=1):
        
        super(Dense_layer, self).__init__()
        ## should we need it?
        if isinstance(in_dim, (tuple, list)) and len(in_dim) > 1:
            in_dim = np.prod(in_dim)
        ###
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.w, self.runtime_coeff = get_weight([out_dim, in_dim], gain=gain, use_wscale=use_wscale, lrmul=lrmul) # use runtime_coeff in dense will break?
        self.w = Parameter(torch.randn(out_dim, in_dim).div_(lrmul))
        self.b = None
        self.fused_op = None
        self.scale = (1 / math.sqrt(in_dim)) * lrmul
        self.lrmul = lrmul
    
    def apply_bias_act(self, fused_op=False, act='linear', bias_init=0, alpha=None):
        # only support leakyrelu for fused_op now.
        self.b = Parameter(torch.zeros(self.out_dim).fill_(bias_init))
        self.act = act
        self.alpha = alpha
        self.fused_op = fused_op
    
    def forward(self, x):
        assert x.shape[1] == self.in_dim, "unmatched shape. {x.shape[1]} v.s. {self.in_dim}"
        
        x =  F.linear(x, self.w) * self.scale
        if self.fused_op:
            x = fused_leaky_relu(x, self.b * self.lrmul)
            return x 
        else:
            if self.b is not None:
                x = torch.add(x, self.b * self.lrmul)
                return activation_funcs[self.act](x, alpha=self.alpha)
            else:
                return x

            
class Conv2d_layer(nn.Module):
    def __init__(self,
                 in_channel, out_channel,
                 kernel=3, mode=None, use_bias=True,
                 resample_kernel=[1,3,3,1],
                 gain=1, use_wscale=True, lrmul=1):
        super(Conv2d_layer, self).__init__()
        assert mode in ['up', 'down', None]
        
        self.out_channel, self.kernel = out_channel, kernel
        self.mode = mode
        self.use_bias = use_bias
        self.padding = kernel // 2
        
        if self.mode == 'up':
            factor = 2
            p = (len(resample_kernel) - factor) - (kernel - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(resample_kernel, pad=(pad0, pad1), upsample_factor=factor)
        elif self.mode == 'down':
            factor = 2
            p = (len(resample_kernel) - factor) + (kernel - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(resample_kernel, pad=(pad0, pad1))
        
        self.w, self.runtime_coeff = get_weight([out_channel, in_channel, kernel, kernel],
                                                gain=gain, use_wscale=use_wscale, lrmul=lrmul)
        
        if use_bias:
            self.bias = Parameter(torch.zeros(1, out_channel, 1, 1))
                             
    def forward(self, x, *args):
        weight = self.w * self.runtime_coeff
        if self.mode == 'up':
            weight = weight.permute(1,0,2,3)
            x = F.conv_transpose2d(x, weight, stride=2, padding=0)
        elif self.mode == 'down':
            x = self.blur(x)
            x = F.conv2d(x, weight, stride=2, padding=0)
        else:
            x = F.conv2d(x, weight, padding=self.padding)
        
        if self.mode == 'up':
            return self.blur(x)
        
        if self.use_bias:
            return x + self.bias
        return x

    
class Modulated_conv2d_layer(nn.Module):
    def __init__(self,
                 in_channel, out_channel,
                 dlatents_dim,
                 kernel=3, mode=None,
                 demodulate=True, resample_kernel=[1,3,3,1],
                 gain=1, use_wscale=True, lrmul=1):
        super(Modulated_conv2d_layer, self).__init__()
        assert mode in ['up', 'down', None]
        
        self.in_channel, self.out_channel, self.kernel = in_channel, out_channel, kernel
        self.dlatents_dim = dlatents_dim
        self.mode = mode
        self.demodulate = demodulate
        self.padding = kernel // 2
        
        if self.mode == 'up':
            factor = 2
            p = (len(resample_kernel) - factor) - (kernel - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(resample_kernel, pad=(pad0, pad1), upsample_factor=factor)
        elif self.mode == 'down':
            factor = 2
            p = (len(resample_kernel) - factor) + (kernel - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(resample_kernel, pad=(pad0, pad1))
        
        self.w, self.runtime_coeff = get_weight([1, out_channel, in_channel, kernel, kernel],
                                                gain=gain, use_wscale=use_wscale, lrmul=lrmul)
        self.scale = 1 / math.sqrt(in_channel * kernel ** 2)
        self.dense = Dense_layer(dlatents_dim, in_channel)
        self.dense.apply_bias_act(bias_init=1)
        
    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel}, '
            f'mode={self.mode})'
        )        
                             
    def forward(self, x, dlatents_in):
        # print("(modconv) \t x: ", x.shape, "\t dlatents: ", dlatents_in.shape)
        b, in_channel, h, w = x.shape
        style = self.dense(dlatents_in)
        
        # modulation
        weight = self.w.repeat(int(style.shape[0]), 1, 1, 1, 1)
        weight = weight * style[:, np.newaxis, :, np.newaxis, np.newaxis] * self.scale# * self.runtime_coeff
        
        # Demodulate.
        if self.demodulate:
            d = torch.rsqrt(torch.sum(weight**2, dim=(2,3,4)) + 1e-8)
            weight = weight * d[:, :, np.newaxis, np.newaxis, np.newaxis]
            

        x = x.view(1, -1, h, w)
      
        if self.mode == 'up':
            weight = weight.permute(0,2,1,3,4).reshape(-1, self.out_channel, self.kernel, self.kernel)
            x = F.conv_transpose2d(x, weight, stride=2, padding=0, groups=b)
        elif self.mode == 'down':
            x = self.blur(x)
            weight = weight.view(b * self.out_channel, -1, self.kernel, self.kernel)
            x = F.conv2d(x, weight, stride=2, padding=0, groups=b)
        else:
            #print("Before shape: ", weight.shape)
            # weight = weight.permute(0,2,1,3,4).reshape(-1, self.out_channel, self.kernel, self.kernel).permute(1,0,2,3)
            weight = weight.view(b * self.out_channel, -1, self.kernel, self.kernel)
            #print("x: ", x.shape)
            #print("after shape: ", weight.shape)
            #print("-"*29)
            x = F.conv2d(x, weight, padding=self.padding, groups=b)
        
        x = x.view(b, self.out_channel, x.shape[2], x.shape[3])
        
        if self.mode == 'up':
            return self.blur(x)
            
        return x


class NoiseInjection(nn.Module):
    def __init__(self):
        super(NoiseInjection, self).__init__()

        self.nosie_stength = Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None: # random noise
            batch, _, height, width = x.shape
            noise = x.new_empty(batch, 1, height, width).normal_()

        return x + self.nosie_stength * noise
            

class Layer(nn.Module):
    """ Layer capsulates modulate convolution layer,
        nonlinearilty and noise layer.
    """
    def __init__(self,
                 in_channel, out_channel,
                 use_modulate=False, dlatents_dim=None,
                 **kwargs):
        super(Layer, self).__init__()
        assert isinstance(use_modulate, bool)
        assert not(use_modulate and not isinstance(dlatents_dim, int)), "dlatent_dim is required when using mod_conv"
        self.use_modulate = use_modulate
        self.noise = None
        if use_modulate:
            self.conv = Modulated_conv2d_layer(in_channel, out_channel, dlatents_dim, **kwargs)
            self.noise = NoiseInjection()
        else:
            self.conv = Conv2d_layer(in_channel, out_channel, **kwargs)
        
        self.act = FusedLeakyReLU(out_channel)
        
    def forward(self, latents, dlatents=None, noise=None):
        # print("(Layer) \t latents: ", latents.shape, "\t dlatents: ", dlatents.shape)
        if self.use_modulate and dlatents is None:
            raise RuntimeError("modulate conv needs dlatents(style) input")
        x = self.conv(latents, dlatents)
        if self.noise is not None:
            x = self.noise(x, noise=noise)
        return self.act(x)
        
        
class ToRGB(nn.Module):
    def __init__(self, in_channel, dlatents_dim, up=True, resample_kernel=[1, 3, 3, 1]):
        super(ToRGB, self).__init__()

        if up:
            self.upsample = Upsample(resample_kernel)

        self.conv = Modulated_conv2d_layer(in_channel, 3, dlatents_dim, kernel=1, demodulate=False)
        self.bias = Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, latent, style, skip=None):
        x = self.conv(latent, style)
        x = x + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            x = x + skip
        return x
    

class FromRGB(nn.Module):
    def __init__(self, out_channel, resample_kernel=[1, 3, 3, 1]):
        super(FromRGB, self).__init__()
        
        self.conv = Conv2d_layer(3, out_channel, kernel=1)
        self.bias = Parameter(torch.zeros(1, out_channel, 1, 1))
        self.act = FusedLeakyReLU(out_channel)
        
    def forward(self, rgb_in):
        out = self.conv(rgb_in) + self.bias
        return self.act(out)


def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    b, c, h, w = x.shape
    group_size = np.minimum(group_size, b)
    y = x.reshape(group_size, -1, num_new_features, c//num_new_features, h, w).float()
    y = y - y.mean(dim=0, keepdims=True)
    y = torch.sqrt(torch.mean(y**2, dim=0) + 1e-8)
    y = torch.mean(y, dim=[2,3,4], keepdims=True)
    y = torch.mean(y, dim=2)
    y = y.type(x.dtype)
    y = y.repeat(group_size, 1, h, w)
    return torch.cat([x, y], dim=1)
    

class DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, architecture, resample_kernel=[1, 3, 3, 1], **kwargs):
        super(DBlock, self).__init__()
        assert architecture in ['skip', 'resnet'], "unsupoorted D. type"
        self.architecture = architecture
        if architecture == 'skip':
            self.frgb = FromRGB(in_channel)
        else:
            self.skip = Conv2d_layer(in_channel, out_channel, kernel=1)
        
        self.scale = Downsample(resample_kernel)
        self.conv = Layer(in_channel, in_channel, resample_kernel=resample_kernel)
        self.conv_down = Layer(in_channel, out_channel, mode='down', resample_kernel=resample_kernel)
    
    def forward(self, latents_in, skip=None):
        if skip is not None:
            y = self.frgb(skip)
            if latents_in is not None:
                y = (y + latents_in) 
        else:
            # resnet
            y = latents_in
        if latents_in is not None:
            x = self.conv(y)
            out = self.conv_down(x)
        skip_out = self.scale(y)
        
        if self.architecture == 'resnet':
            return self.skip(skip_out) + out, None
        return out, skip_out
            
        
class G_synthesis_stylegan2(nn.Module):
    def __init__(self,
        num_layers,
        resolution_log2,
        dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
        num_channels        = 3,            # Number of output color channels.
        kernel = 3,
        fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        fmap_min            = 1,            # Minimum number of feature maps in any layer.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
        **_kwargs):                         # Ignore unrecognized keyword args.)
        
        
        super(G_synthesis_stylegan2, self).__init__()
        self.resolution_log2 = resolution_log2
        self.num_layers = num_layers
        def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

        act = nonlinearity
        self.architecture = architecture
        if self.architecture in ['skip', 'resnet']:
            self.trgbs = nn.ModuleList()
        
        # Prepare noise inputs.
        if randomize_noise is False:
            for layer_idx in range(self.num_layers):
                res = (layer_idx + 5) // 2
                shape = [1, 1, 2 ** res, 2 ** res]
                self.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))
            
        # 4x4
        self.input = Parameter(torch.randn((1, nf(1), 4, 4)))
        self.bottom_layer = Layer(nf(1), nf(1), use_modulate=True, dlatents_dim=dlatent_size, kernel=kernel, resample_kernel=resample_kernel)
        self.trgbs.append(ToRGB(nf(1), dlatent_size))
        
        # main layers
        self.convs = nn.ModuleList()
        
        in_channel = nf(1)
        for res in range(3, self.resolution_log2 + 1):
            fmaps = nf(res-1)
            self.convs.extend([
                Layer(in_channel, fmaps, use_modulate=True, dlatents_dim=dlatent_size, kernel=kernel, mode='up', resample_kernel=resample_kernel),
                Layer(fmaps, fmaps, use_modulate=True, dlatents_dim=dlatent_size, kernel=kernel, resample_kernel=resample_kernel)
            ])
            if self.architecture == 'skip':
                self.trgbs.append(ToRGB(fmaps, dlatent_size))
            in_channel = fmaps

    def forward(self, dlatents_in):
        noises = [getattr(self, f'noise_{i}', None) for i in range(self.num_layers)]
        tile_in = self.input.repeat(dlatents_in.shape[0], 1, 1, 1)
        
        x = self.bottom_layer(tile_in, dlatents_in[:, 0], noise=noises[0])
        if self.architecture == 'skip':
            skip = self.trgbs[0](x, dlatents_in[:, 1])
        
        # main layer
        for res in range(3, self.resolution_log2 + 1):
            x = self.convs[res*2 - 6](x, dlatents_in[:, res*2-5], noises[res*2-5])
            x = self.convs[res*2 - 5](x, dlatents_in[:, res*2-4], noises[res*2-4])
            # Does the style of ToRGB is shared with the next conv layer?
            if self.architecture == 'skip' or res == self.resolution_log2:
                skip = self.trgbs[res-2](x, dlatents_in[:, res*2-3], skip=skip)
        images_out = skip
        
        return images_out
            
            
class G_mapping(nn.Module):
    def __init__(self,
        latent_size             = 512,          # Latent vector (Z) dimensionality.
        label_size              = 0,            # Label dimensionality, 0 if no labels.
        dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
        mapping_layers          = 8,            # Number of mapping layers.
        mapping_fmaps           = 512,          # Number of activations in the mapping layers.
        mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
        normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
        **_kwargs):                             # Ignore unrecognized keyword args.
        
        super(G_mapping, self).__init__()
        assert isinstance(label_size, int) and label_size >= 0
        
        self.label_size = label_size
        self.mapping_fmaps = mapping_fmaps
        self.normalize_latents = normalize_latents
    
        
        if label_size > 0:
            self.embedding = nn.Embedding(label_size, latent_size)
        fan_in = 2 * latent_size if label_size>0 else latent_size
        fc = []
        for layer_idx in range(mapping_layers):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            d = Dense_layer(fan_in, fmaps, lrmul=mapping_lrmul)
            d.apply_bias_act(fused_op=True)
            fc.append(d)
            fan_in = fmaps
        self.fc = nn.Sequential(*fc)
        
        # self.init_weights()
            
    # def init_weights(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
    #             nn.init.normal_(module.weight, 0, 1) # fix init std.
            
    def forward(self, latents, labels=None, dlatent_broadcast=None):
        x = latents
        if self.label_size > 0:
            assert len(labels.shape) == 1
            y = self.embedding(labels)
            x = torch.cat((x, y), dim=1)
            
        if self.normalize_latents:
            x = x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)
            
        x = self.fc(x)
        if dlatent_broadcast:
            x = x.unsqueeze(1).repeat(1, dlatent_broadcast, 1) 
        dlatents_out = x
        return dlatents_out
