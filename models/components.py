import math
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torch_utils.ops import upfirdn2d, FusedLeakyReLU, fused_leaky_relu


_activation_funcs = {
    'linear': lambda x, **_: x,
    'fused_lrelu': lambda x, b, **_: fused_leaky_relu(x, b)
}


# from: https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()
    return k


def get_weight(shape, gain=1, use_wscale=True, lrmul=1):
    fan_in = np.prod(shape[1:])  # [fmaps_out, fmaps_in, kernel, kernel] or [out, in]
    he_std = gain / np.sqrt(fan_in)  # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    return Parameter(torch.randn(*shape).mul_(init_std)), runtime_coef


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)


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
        return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)


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
        return upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)


class Dense_layer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bias=True,
        bias_init=0,
        nonlinearity='fused_lrelu',
        alpha=0.2,
        gain=1,
        use_wscale=True,
        lrmul=1,
    ):
        super(Dense_layer, self).__init__()
        if isinstance(in_dim, (tuple, list)) and len(in_dim) > 1:
            in_dim = np.prod(in_dim)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias

        self.w, self.runtime_coeff = get_weight([out_dim, in_dim], gain=gain, use_wscale=use_wscale, lrmul=lrmul)
        # self.w = Parameter(torch.randn(out_dim, in_dim).div_(lrmul))
        # self.scale = (1 / math.sqrt(in_dim)) * lrmul
        self.lrmul = lrmul

        if use_bias:
            self.b = Parameter(torch.zeros(out_dim).fill_(bias_init))

        self.act = nonlinearity
        self.alpha = alpha

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.out_dim}, {self.in_dim} '
                f'bias={self.use_bias}, act={self.act})')

    def forward(self, x):
        assert x.shape[1] == self.in_dim, f"unmatched shape. {x.shape[1]} v.s. {self.in_dim}"

        x = F.linear(x, self.w * self.runtime_coeff)
        if self.act.startswith('fused'):
            return _activation_funcs[self.act](x, self.b * self.lrmul)

        if self.use_bias:
            x.add_(self.b * self.lrmul)

        return _activation_funcs[self.act](x, alpha=self.alpha)


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class Conv2d_layer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel=3,
        mode=None,
        use_bias=True,
        resample_kernel=[1, 3, 3, 1],
        gain=1,
        use_wscale=True,
        lrmul=1
    ):
        super(Conv2d_layer, self).__init__()
        assert mode in ['up', 'down', None]

        self.out_channel, self.in_channel, self.kernel = out_channel, in_channel, kernel
        self.mode = mode
        self.padding = kernel // 2
        self.use_bias = use_bias

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
            self.bias_act = FusedLeakyReLU(out_channel)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.out_channel}, {self.in_channel}, {self.kernel} '
                f'mode={self.mode}')

    def forward(self, x, *args):
        weight = self.w * self.runtime_coeff
        if self.mode == 'up':
            weight = weight.permute(1, 0, 2, 3)
            x = F.conv_transpose2d(x, weight, stride=2, padding=0)
        elif self.mode == 'down':
            x = self.blur(x)
            x = F.conv2d(x, weight, stride=2, padding=0)
        else:
            x = F.conv2d(x, weight, padding=self.padding)

        if self.mode == 'up':
            return self.blur(x)

        if self.use_bias:
            return self.bias_act(x)

        return x


class Modulated_conv2d_layer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        dlatents_dim,
        kernel=3,
        mode=None,
        demodulate=True,
        resample_kernel=[1, 3, 3, 1],
        gain=1,
        use_wscale=True,
        lrmul=1
    ):
        super(Modulated_conv2d_layer, self).__init__()
        assert mode in ['up', 'down', None]

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dlatents_dim = dlatents_dim
        self.kernel = kernel
        self.mode = mode
        self.demodulate = demodulate
        self.padding = kernel // 2

        if self.mode == 'up':
            factor = 2
            p = (len(resample_kernel) - factor) - (kernel - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(resample_kernel, pad=(pad0, pad1),
                             upsample_factor=factor)
        elif self.mode == 'down':
            factor = 2
            p = (len(resample_kernel) - factor) + (kernel - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(resample_kernel, pad=(pad0, pad1))

        self.w, self.runtime_coeff = get_weight(
            [out_channel, in_channel, kernel, kernel],
            gain=gain, use_wscale=use_wscale, lrmul=lrmul
        )
        self.scale = 1 / math.sqrt(in_channel * kernel ** 2)
        self.dense = Dense_layer(dlatents_dim, in_channel, bias_init=1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.out_channel}, '
                f'{self.in_channel}, {self.kernel}, '
                f'mode={self.mode}, demodulate={self.demodulate})')

    def forward(self, x, dlatents_in):
        # print("(modconv) \t x: ", x.shape, "\t dlatents: ", dlatents_in.shape)
        b, in_channel, h, w = x.shape
        style = self.dense(dlatents_in)

        # modulation
        weight = self.w * self.scale
        weight = weight.unsqueeze(0).repeat(int(style.shape[0]), 1, 1, 1, 1)
        weight = weight * style[:, np.newaxis, :, np.newaxis, np.newaxis]

        if self.demodulate:
            d = torch.rsqrt(torch.sum(weight**2, dim=(2, 3, 4)) + 1e-8)
            weight = weight * d[:, :, np.newaxis, np.newaxis, np.newaxis]

        x = x.view(1, -1, h, w)

        if self.mode == 'up':
            weight = weight.permute(0, 2, 1, 3, 4)
            weight = weight.reshape(-1, self.out_channel, self.kernel, self.kernel)
            x = F.conv_transpose2d(x, weight, stride=2, padding=0, groups=b)
        elif self.mode == 'down':
            x = self.blur(x)
            weight = weight.view(b * self.out_channel, -1, self.kernel, self.kernel)
            x = F.conv2d(x, weight, stride=2, padding=0, groups=b)
        else:
            weight = weight.view(b * self.out_channel, -1, self.kernel, self.kernel)
            x = F.conv2d(x, weight, padding=self.padding, groups=b)

        x = x.view(b, self.out_channel, x.shape[2], x.shape[3])

        if self.mode == 'up':
            return self.blur(x)

        return x


class NoiseInjection(nn.Module):
    def __init__(self):
        super(NoiseInjection, self).__init__()
        self.noise_strength = Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            # random noise
            batch, _, height, width = x.shape
            noise = x.new_empty(batch, 1, height, width).normal_()

        return x + self.noise_strength * noise


class Layer(nn.Module):
    """ Layer capsulates modulate convolution layer,
        nonlinearilty and noise layer.
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        use_modulate=False,
        dlatents_dim=None,
        **kwargs,
    ):
        super(Layer, self).__init__()
        assert isinstance(use_modulate, bool)
        assert not(use_modulate and not isinstance(dlatents_dim, int)), \
            "dlatent_dim is required when using mod_conv"
        self.use_modulate = use_modulate
        self.noise = None
        if use_modulate:
            self.conv = Modulated_conv2d_layer(in_channel, out_channel, dlatents_dim, **kwargs)
            self.noise = NoiseInjection()
        else:
            self.conv = Conv2d_layer(in_channel, out_channel, **kwargs)

        self.act = FusedLeakyReLU(out_channel)

    def forward(self, latents, dlatents=None, noise=None):
        if self.use_modulate and dlatents is None:
            raise RuntimeError("modulate conv needs dlatents(style) input")

        x = self.conv(latents, dlatents)

        if self.noise is not None:
            x = self.noise(x, noise=noise)

        return self.act(x)


class ContentEncoder(nn.Module):
    def __init__(self, res_log2, res_out_log2, nf_in, max_nf):
        super(ContentEncoder, self).__init__()
        convs = []
        for i in range(res_out_log2, res_log2):
            nf_out = min(64 * 2 ** (i - res_out_log2 + 1), max_nf)
            convs.append(Conv2d_layer(nf_in, nf_out, mode='down'))
            nf_in = nf_out
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        outs = []
        for conv in self.convs:
            x = conv(x)
            outs.append(x)

        return outs


def style_encoder(res_log2, res_out_log2, nf_in, max_nf):
    encoding_list = []
    for i in range(res_out_log2, res_log2):
        nf_out = min(64 * (i + 1), max_nf)
        encoding_list.extend(
            [Conv2d_layer(nf_in, nf_out, mode='down')])
        nf_in = nf_out
    encoding_list.append(nn.AdaptiveAvgPool2d((1, 1)))

    return nn.Sequential(*encoding_list)


class ToRGB(nn.Module):
    def __init__(self, in_channel, out_channel, dlatents_dim, up=True, resample_kernel=[1, 3, 3, 1]):
        super(ToRGB, self).__init__()

        if up:
            self.upsample = Upsample(resample_kernel)

        self.conv = Modulated_conv2d_layer(in_channel, out_channel, dlatents_dim, kernel=1, demodulate=False)
        self.bias = Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, latent, style, skip=None):
        x = self.conv(latent, style)
        x = x + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            x = x + skip

        return x


class FromRGB(nn.Module):
    def __init__(self, in_channel, out_channel, resample_kernel=[1, 3, 3, 1]):
        super(FromRGB, self).__init__()

        self.conv = Conv2d_layer(in_channel, out_channel, kernel=1)
        # self.bias = Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.act = FusedLeakyReLU(out_channel)

    def forward(self, rgb_in):
        return self.conv(rgb_in)


def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    b, c, h, w = x.shape
    group_size = np.minimum(group_size, b)
    y = x.reshape(group_size, -1, num_new_features, c // num_new_features, h, w).float()
    y = y - y.mean(dim=0, keepdims=True)
    y = torch.sqrt(torch.mean(y**2, dim=0) + 1e-8)
    y = torch.mean(y, dim=[2, 3, 4], keepdims=True).squeeze(2)
    y = y.type(x.dtype)
    y = y.repeat(group_size, 1, h, w)
    return torch.cat([x, y], dim=1)


class DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, architecture,
                 resample_kernel=[1, 3, 3, 1], **kwargs):
        super(DBlock, self).__init__()
        assert architecture in ['skip', 'resnet'], "unsupoorted D. type"
        self.architecture = architecture
        if architecture == 'skip':
            self.frgb = FromRGB(in_channel)
        else:
            self.skip = Conv2d_layer(in_channel, out_channel, kernel=1, use_bias=False)

        self.scale = Downsample(resample_kernel)
        self.conv = Layer(in_channel, in_channel, resample_kernel=resample_kernel, use_bias=False)
        self.conv_down = Layer(in_channel, out_channel, mode='down',
                               resample_kernel=resample_kernel, use_bias=False)

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
    def __init__(
        self,
        num_layers,
        resolution_log2,
        dlatent_size=512,            # Disentangled latent (W) dimensionality.
        num_channels=3,              # Number of output color channels.
        kernel=3,
        fmap_base=16 << 10,          # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,              # log2 feature map reduction when doubling the resolution.
        fmap_min=1,                  # Minimum number of feature maps in any layer.
        fmap_max=512,                # Maximum number of feature maps in any layer.
        randomize_noise=True,        # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        architecture='skip',         # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity='lrelu',        # Activation function: 'relu', 'lrelu', etc.
        resample_kernel=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
        **_kwargs,                     # Ignore unrecognized keyword args.)
    ):

        super(G_synthesis_stylegan2, self).__init__()
        self.resolution_log2 = resolution_log2
        self.num_layers = num_layers

        def nf(stage):
            return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

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
        content_ch = 4
        face_style_dim = 112
        dlatent_size += face_style_dim
        self.input = Parameter(torch.randn((1, nf(1), 4, 4)))
        self.style_encoder = style_encoder(resolution_log2, 2, 3, face_style_dim)
        self.ContentEncoder = ContentEncoder(resolution_log2, 2, content_ch, nf(1) // 2)
        self.bottom_layer = Layer(int(nf(1) * 1.5), int(nf(1) * 1.5), use_modulate=True, dlatents_dim=dlatent_size, kernel=kernel, resample_kernel=resample_kernel)
        self.trgbs.append(ToRGB(int(nf(1) * 1.5), num_channels, dlatent_size))

        # main layers
        self.convs = nn.ModuleList()

        in_channel = int(nf(1) * 1.5)
        for res in range(3, self.resolution_log2 + 1):
            fmaps = int(nf(res - 1) * 1.5)
            fmaps1 = fmaps if res == self.resolution_log2 else nf(res - 1)
            self.convs.extend([
                Layer(in_channel, fmaps1, use_modulate=True, dlatents_dim=dlatent_size, kernel=kernel, mode='up', resample_kernel=resample_kernel),
                Layer(fmaps, fmaps, use_modulate=True, dlatents_dim=dlatent_size, kernel=kernel, resample_kernel=resample_kernel)
            ])
            if self.architecture == 'skip':
                self.trgbs.append(ToRGB(fmaps, num_channels, dlatent_size))
            in_channel = fmaps

    def forward(self, dlatents_in, style_in=None, content_in=None):
        noises = [getattr(self, f'noise_{i}', None) for i in range(self.num_layers)]
        tile_in = self.input.repeat(dlatents_in.shape[0], 1, 1, 1)
        content_encoding = self.ContentEncoder(content_in)
        style_encoding = self.style_encoder(style_in)
        tile_in = torch.cat([tile_in, content_encoding[-1]], dim=1)

        style_encoding = style_encoding.squeeze().unsqueeze(1).repeat(1, dlatents_in.shape[1], 1)
        dlatents_in = torch.cat([style_encoding, dlatents_in], dim=2)

        x = self.bottom_layer(tile_in, dlatents_in[:, 0], noise=noises[0])
        if self.architecture == 'skip':
            skip = self.trgbs[0](x, dlatents_in[:, 1])

        # main layer
        for res in range(3, self.resolution_log2 + 1):
            x = self.convs[res * 2 - 6](x, dlatents_in[:, res * 2 - 5], noises[res * 2 - 5])
            if res < self.resolution_log2:
                x = torch.cat([x, content_encoding[7 - res]], dim=1)
            x = self.convs[res * 2 - 5](x, dlatents_in[:, res * 2 - 4], noises[res * 2 - 4])

            if self.architecture == 'skip' or res == self.resolution_log2:
                skip = self.trgbs[res - 2](x, dlatents_in[:, res * 2 - 3], skip=skip)
        images_out = skip

        return images_out


class G_mapping(nn.Module):
    def __init__(
        self,
        latent_size=512,         # Latent vector (Z) dimensionality.
        label_size=0,            # Label dimensionality, 0 if no labels.
        embedding_size=0,
        dlatent_size=512,        # Disentangled latent (W) dimensionality.
        mapping_layers=8,        # Number of mapping layers.
        mapping_fmaps=512,       # Number of activations in the mapping layers.
        mapping_lrmul=0.01,      # Learning rate multiplier for the mapping layers.
        normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
        **_kwargs                # Ignore unrecognized keyword args.
    ):
        super(G_mapping, self).__init__()
        assert isinstance(label_size, int) and label_size >= 0

        self.label_size = label_size
        self.mapping_fmaps = mapping_fmaps
        self.normalize_latents = normalize_latents

        if label_size > 0:
            self.embedding = nn.Embedding(label_size, embedding_size)
        fan_in = (embedding_size + latent_size) if label_size > 0 else latent_size
        fc = []
        for layer_idx in range(mapping_layers):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            fc.append(Dense_layer(fan_in, fmaps, lrmul=mapping_lrmul))
            fan_in = fmaps
        self.fc = nn.Sequential(*fc)

        # self.init_weights()

    # def init_weights(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
    #             nn.init.normal_(module.weight, 0, 1) # fix init std.

    def forward(self, latents, labels=None, dlatent_broadcast=None):
        x = latents
        if self.label_size > 0 and len(labels.shape) == 1:
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
