import random
import importlib
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.cuda.amp import autocast

from torch_utils.ops import upfirdn2d, fused_act, conv2d_resample


class Dense_layer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bias=True,
        bias_init=0,
        lrmul=1,
    ):
        super(Dense_layer, self).__init__()
        if isinstance(in_dim, (tuple, list)) and len(in_dim) > 1:
            in_dim = np.prod(in_dim)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias

        self.weight = Parameter(torch.randn(out_dim, in_dim).div_(lrmul))
        self.weight_gain = 1 / np.sqrt(in_dim)
        self.lrmul = lrmul

        self.bias = Parameter(torch.full([out_dim], np.float32(bias_init))) if use_bias else torch.empty([0])

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_dim}, {self.out_dim}, bias={self.use_bias}')

    def forward(self, x):
        assert x.shape[1] == self.in_dim, f"unmatched shape: {x.shape[1]} v.s {self.in_dim}"
        w = self.weight.to(x.dtype) * self.weight_gain * self.lrmul
        b = self.bias * self.lrmul

        x = F.linear(x, w)

        return fused_act.fused_leaky_relu(x, b)


class Conv2d_layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        mode=None,
        use_bias=True,
        resample_filter=[1, 3, 3, 1],
    ):
        super(Conv2d_layer, self).__init__()
        assert mode in ['up', 'down', None]

        self.mode = mode
        self.up = 2 if self.mode == "up" else 1
        self.down = 2 if self.mode == "down" else 1
        self.padding = kernel // 2

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.weight = Parameter(torch.randn([out_channels, in_channels, kernel, kernel]))
        self.weight_gain = 1 / np.sqrt(in_channels * kernel ** 2)
        self.bias = Parameter(torch.zeros([out_channels])) if use_bias else torch.empty([0])

    def forward(self, x, *args):
        weight = self.weight * self.weight_gain
        flip_weight = (self.up == 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=self.resample_filter,
                                            up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        return fused_act.fused_leaky_relu(x, self.bias.to(x.device))


class Modulated_conv2d_layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dlatent_dim,
        kernel=3,
        mode=None,
        demodulate=True,
        resample_filter=[1, 3, 3, 1],
    ):
        super(Modulated_conv2d_layer, self).__init__()
        assert mode in ['up', 'down', None]

        self.mode = mode
        self.dlatent_dim = dlatent_dim
        self.demodulate = demodulate
        self.padding = kernel // 2

        self.up = 2 if self.mode == 'up' else 1
        self.down = 2 if self.mode == 'down' else 1

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.weight = Parameter(torch.randn([out_channels, in_channels, kernel, kernel]))  # already contiguous
        self.scale = 1 / np.sqrt(in_channels * kernel * kernel)
        self.dense = Dense_layer(dlatent_dim, in_channels, bias_init=1)

    def forward(self, x, dlatents_in, noise=None):
        b, in_channels, h, w = x.shape
        style = self.dense(dlatents_in)

        # modulation
        weight = self.weight * self.scale
        weight = weight.unsqueeze(0)
        weight = weight * style.reshape(b, 1, -1, 1, 1)

        if self.demodulate:
            d = torch.rsqrt(torch.sum(weight ** 2, dim=(2, 3, 4)) + 1e-8)
            weight = weight * d.reshape(b, -1, 1, 1, 1)

        # using group conv. to deal instance-wise conv.
        x = x.view(1, -1, h, w)
        weight = weight.reshape(-1, in_channels, *weight.shape[3:])
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=self.resample_filter,
                                            up=self.up, down=self.down, padding=self.padding, groups=b, flip_weight=(self.up == 1))
        x = x.view(b, -1, *x.shape[2:])
        return x


class Layer(nn.Module):
    """ Layer capsulates modulate convolution layer,
        nonlinearilty and noise layer.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        dlatent_dim,
        resolution,
        **kwargs,
    ):
        super(Layer, self).__init__()
        self.resolution = resolution
        self.conv = Modulated_conv2d_layer(in_channels, out_channels, dlatent_dim, **kwargs)

        self.noise_strength = Parameter(torch.zeros([1]))
        self.register_buffer('noise_const', torch.randn([1, 1, resolution, resolution]))  # only using by G_ema
        self.bias = Parameter(torch.zeros([out_channels]))

    def forward(self, latents, dlatents, noise_mode='random'):
        assert noise_mode in ['random', 'const']
        x = self.conv(latents, dlatents)

        noise = None
        if noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        if noise is not None:
            x = x.add_(noise)

        return fused_act.fused_leaky_relu(x, self.bias)


class ContentEncoder(nn.Module):
    def __init__(self, res_in, res_out, nf_in=3, max_nf=512):
        assert (res_in / res_out) % 2 == 0
        super(ContentEncoder, self).__init__()
        num_layer = int(np.log2(res_in / res_out))
        self.num_layer = num_layer

        res = res_in
        for i in range(num_layer):
            nf_out = min(2 ** (i + 7), max_nf)  # to get half channels from synthesis layers
            res = res // 2
            setattr(self, f'b{res}', Conv2d_layer(nf_in, nf_out, mode='down'))
            nf_in = nf_out
        assert res == res_out, res

    def forward(self, x):
        res = x.shape[2]
        outs = {}
        for i in range(self.num_layer):
            res = res // 2
            conv_down = getattr(self, f'b{res}')
            x = conv_down(x)
            outs[f'b{res}'] = x

        return outs


def get_encoder(res_in, res_out, nf_in=3, max_nf=512, dlatent_dim=512):
    assert (res_in / res_out) % 2 == 0
    num_layer = int(np.log2(res_in / res_out))

    enc = []
    for i in range(num_layer):
        nf_out = min(2 ** (i + 3), max_nf)
        enc.append(Conv2d_layer(nf_in, nf_out, mode='down'))
        nf_in = nf_out
    enc.append(Conv2d_layer(nf_in, dlatent_dim, kernel=1))
    enc.append(nn.AdaptiveAvgPool2d((1, 1)))

    return nn.Sequential(*enc)


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, dlatent_dim, resample_filter=[1, 3, 3, 1]):
        super(ToRGB, self).__init__()
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.conv = Modulated_conv2d_layer(in_channels, out_channels, dlatent_dim, kernel=1, demodulate=False)
        self.bias = Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, latent, style, skip=None):
        x = self.conv(latent, style)
        x = x + self.bias

        if skip is not None:
            skip = upfirdn2d.upsample2d(x=skip, f=self.resample_filter, up=2)
            x = x + skip

        return x


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
    def __init__(self, in_channels, out_channels,
                 architecture='resnet', resample_filter=[1, 3, 3, 1], **kwargs):
        super(DBlock, self).__init__()
        assert architecture in ['skip', 'resnet'], "unsupoorted D. type"
        self.architecture = architecture
        if architecture == 'skip':
            self.frgb = Conv2d_layer(in_channels, out_channels, kernel=1)
        else:
            self.skip = Conv2d_layer(in_channels, out_channels, kernel=1, use_bias=False)

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.conv = Conv2d_layer(in_channels, in_channels, resample_filter=resample_filter)
        self.conv_down = Conv2d_layer(in_channels, out_channels, mode='down', resample_filter=resample_filter)

    def forward(self, latents_in, skip=None):
        if skip is not None:
            assert self.architecture == 'skip'
            y = self.frgb(skip)
            y = (y + latents_in) if latents_in is not None else y
        else:
            # resnet
            y = latents_in
        if latents_in is not None:
            x = self.conv(y)
            out = self.conv_down(x)
        skip_out = upfirdn2d.downsample2d(x=y, f=self.resample_filter, down=2)

        if self.architecture == 'resnet':
            return self.skip(skip_out) + out, None
        return out, skip_out


class G_synthesis_stylegan2(nn.Module):
    def __init__(
        self,
        dlatent_dim,                 # Disentangled latent (W) dimensionality.
        img_resolution,              # Output resolution
        img_channels=3,              # Number of output color channels.
        fmap_base=16384,             # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,              # log2 feature map reduction when doubling the resolution.
        fmap_min=1,                  # Minimum number of feature maps in any layer.
        fmap_max=512,                # Maximum number of feature maps in any layer.
        architecture='skip',         # Architecture: 'orig', 'skip', 'resnet'.
        use_content_encoder=False,   # encoding content & concat fmaps to synthesis network (like U-Net)
        content_encoder_kwargs=None,
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
    ):
        super(G_synthesis_stylegan2, self).__init__()
        res_log2 = int(np.log2(img_resolution))
        self.res_log2 = res_log2
        self.resolutions = [2 ** i for i in range(2, res_log2 + 1)]
        self.use_content_encoder = use_content_encoder

        def nf(stage):
            return int(np.clip(fmap_base / (2 ** (stage * fmap_decay)), fmap_min, fmap_max))

        # act = nonlinearity
        self.architecture = architecture

        if use_content_encoder:
            self.ContentEncoder = ContentEncoder(img_resolution, 4, **content_encoder_kwargs)

        # 4x4
        bottom_ch = nf(1) // 2 if use_content_encoder else nf(1)
        self.input = Parameter(torch.randn([1, bottom_ch, 4, 4]))

        # main layers
        self.convs = nn.ModuleList()

        for res_log2, res in enumerate(self.resolutions, 2):
            fmaps = nf(res_log2 - 1)
            fmaps1 = fmaps // 2 if use_content_encoder and res != img_resolution else fmaps

            if res > 4:
                setattr(self, f'b{res}_up', Layer(in_ch, fmaps1, dlatent_dim, res, mode='up', resample_filter=resample_filter))
            setattr(self, f'b{res}', Layer(fmaps, fmaps, dlatent_dim, res, resample_filter=resample_filter))

            if self.architecture == 'skip':
                setattr(self, f'b{res}_trgb', ToRGB(fmaps, img_channels, dlatent_dim, resample_filter=resample_filter))

            in_ch = fmaps

    def forward(self, dlatents_in, content_in=None, **layer_kwargs):
        skip = None
        x = self.input.repeat(dlatents_in.shape[0], 1, 1, 1)

        if self.use_content_encoder:
            with torch.autograd.profiler.record_function("Content encoder"):
                content_encoding = self.ContentEncoder(content_in)

        with torch.autograd.profiler.record_function("Synthesis Main"):
            for res_log2, res in enumerate(self.resolutions, 2):
                if res > 4:
                    conv_up = getattr(self, f'b{res}_up')
                    x = conv_up(x, dlatents_in[:, res_log2 * 2 - 5], **layer_kwargs)

                if self.use_content_encoder and res != self.resolutions[-1]:
                    x = torch.cat([x, content_encoding[f'b{res}']], dim=1)

                x = getattr(self, f'b{res}')(x, dlatents_in[:, res_log2 * 2 - 4], **layer_kwargs)

                if self.architecture == 'skip' or res == self.resolutions[-1]:
                    # with autocast(enabled=False):
                    skip = getattr(self, f'b{res}_trgb')(x, dlatents_in[:, res_log2 * 2 - 3], skip=skip)

            images_out = skip

        return images_out


class G_mapping(nn.Module):
    def __init__(
        self,
        latent_size,             # Latent vector (Z) dimensionality.
        label_size,              # Label dimensionality, 0 if no labels.
        embed_dim=0,
        dlatent_dim=512,         # Disentangled latent (W) dimensionality.
        num_layer=8,             # Number of mapping layers.
        num_channel=512,         # Number of activations in the mapping layers.
        lrmul=0.01,              # Learning rate multiplier for the mapping layers.
        normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
        **_kwargs                # Ignore unrecognized keyword args.
    ):
        assert isinstance(label_size, int) and label_size >= 0
        super(G_mapping, self).__init__()

        self.label_size = label_size
        self.conditional = label_size > 0
        self.normalize_latents = normalize_latents

        if label_size > 0:
            assert embed_dim > 0
            self.embedding = nn.Embedding(label_size, embed_dim)

        fc = []
        in_dim = (embed_dim + latent_size) if label_size > 0 else latent_size
        for layer_idx in range(num_layer):
            out_dim = dlatent_dim if layer_idx == num_layer - 1 else num_channel
            fc.append(Dense_layer(in_dim, out_dim, lrmul=lrmul))
            in_dim = out_dim
        self.fc = nn.Sequential(*fc)

    def forward(self, latents, labels=None, dlatent_broadcast=None):
        x = latents

        if self.conditional:
            assert labels is not None and isinstance(labels, torch.Tensor)
            assert labels.ndim == 1 and labels.size == x.shape[0]
            assert labels.max() < self.label_size, f"labels: {labels}"
            y = self.embedding(labels)
            x = torch.cat((x, y), dim=1)

        if self.normalize_latents:
            x = x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

        dlatent = self.fc(x)

        if dlatent_broadcast:
            return dlatent.unsqueeze(1).repeat(1, dlatent_broadcast, 1)

        return dlatent


class Generator(nn.Module):
    def __init__(
        self,
        z_dim,
        label_size,
        resolution,
        extra_channels=0,
        # mapping_network='G_mapping',
        # synthesis_netowrk='G_synthesis_stylegan2',
        map_kwargs=None,
        use_style_encoder=False,
        style_encoder_kwargs=None,
        synthesis_kwargs=None,
        # use_noise=True,
        # is_training=None,
        # truncation_psi=0.5,
        # truncation_cut_off=None,
        # truncation_psi_val=None,
        # truncation_cut_off_val=None,
        # dlatent_avg_beta=0.995,
        **kwargs
    ):
        assert resolution >= 4 and resolution & (resolution - 1) == 0

        super(Generator, self).__init__()

        # assert is_training is not None
        # if is_training:
        #     truncation_psi = truncation_cut_off = None
        # else: # ignore some condition.
        #     truncation_psi = truncation_psi_val
        #     truncation_cut_off = truncation_cut_off_val
        #     dlatent_avg_beta = None
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.num_channels = 3 + extra_channels
        self.use_style_encoder = use_style_encoder

        self.mapping = G_mapping(z_dim, label_size, **map_kwargs)
        dlatent_dim = map_kwargs.dlatent_dim

        if use_style_encoder:
            assert style_encoder_kwargs is not None
            dlatent_dim += style_encoder_kwargs.dlatent_dim
            self.style_encoder = get_encoder(resolution, 4, **style_encoder_kwargs)

        self.synthesis = G_synthesis_stylegan2(
            dlatent_dim, resolution, img_channels=self.num_channels, architecture='skip', **synthesis_kwargs)

    def forward(self, latents_in, labels_in=None, style_in=None, content_in=None, return_latents=None, **synthesis_kwargs):
        with torch.autograd.profiler.record_function("G Mapping"):
            dlatents = self.get_dlatent(latents_in, labels_in=labels_in)

        if self.use_style_encoder:
            assert style_in is not None
            with torch.autograd.profiler.record_function("Style encoder"):
                style_encoding = self.style_encoder(style_in)
                style_encoding = style_encoding.flatten(1).unsqueeze(1).repeat(1, dlatents.shape[1], 1)  # [N, w_broadcast, w_dim]
                dlatents = torch.cat([style_encoding, dlatents], dim=2)

        with torch.autograd.profiler.record_function("G synthesis"):
            images_out = self.synthesis(dlatents, content_in=content_in, **synthesis_kwargs)

        if return_latents:
            return images_out, dlatents
        return images_out, None

    def get_dlatent(self, latents_in, labels_in=None):
        assert isinstance(latents_in, (tuple, List)), "valid latent dim: [n_noise, B, z_dim]"
        assert len(latents_in) in [1, 2]
        
        if len(latents_in) == 2:
            # style mixing
            pivot = random.randint(1, self.num_layers - 1)
            mixing_chunks = [pivot, self.num_layers - pivot]
            dlatents = [self.mapping(l, labels_in, dlatent_broadcast=i)
                        for l, i in zip(latents_in, mixing_chunks)]
            dlatents = torch.cat(dlatents, dim=1)
        else:
            dlatents = self.mapping(latents_in[0], labels_in, dlatent_broadcast=self.num_layers)

        return dlatents


class Discriminator(nn.Module):
    def __init__(
        self,
        label_size,
        resolution,
        extra_channels=3,
        fmap_base=16 << 10,
        fmap_decay=1.0,
        fmap_min=1,
        fmap_max=512,
        mbstd_group_size=4,
        mbstd_num_features=1,
        resample_kernel=[1, 3, 3, 1],
        architecture='resnet',
        **kwargs,
    ):
        assert architecture in ['skip', 'resnet'], "unsupported D architecture."
        super(Discriminator, self).__init__()
        mbstd_num_channels = 1
        self.img_channels = 3 + extra_channels
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features
        self.resolution_log2 = int(np.log2(resolution))
        self.architecture = architecture

        def nf(stage):
            scaled = int(fmap_base / (2.0 ** (stage * fmap_decay)))
            return np.clip(scaled, fmap_min, fmap_max)

        if self.architecture == 'resnet':
            self.frgb = Conv2d_layer(self.img_channels, nf(self.resolution_log2 - 1), kernel=1)

        self.blocks = nn.ModuleList()
        for res in range(self.resolution_log2, 2, -1):
            self.blocks.append(DBlock(nf(res - 1), nf(res - 2), self.architecture))

        # output layer
        self.conv_out = Conv2d_layer(nf(1) + mbstd_num_channels, nf(1))
        self.dense_out = Dense_layer(512 * 4 * 4, nf(0))
        self.label_out = Dense_layer(nf(0), max(label_size, 1))

    def forward(self, images_in, labels_in=None):
        assert images_in.shape[1] == self.img_channels, f"(D) channel unmatched. {images_in.shape[1]} v.s. {self.img_channels}"

        x = skip = None
        if self.architecture == 'resnet':
            x = self.frgb(images_in)
        else:
            skip = images_in

        for block in self.blocks:
            x, skip = block(x, skip)

        # TODO: FRGB if skip
        x = minibatch_stddev_layer(x, self.mbstd_group_size, self.mbstd_num_features)
        x = self.conv_out(x)
        x = x.view(x.shape[0], -1)
        x = self.dense_out(x)
        out = self.label_out(x)
        if labels_in is not None and labels_in.shape[1] > 0:
            out = torch.mean(out * labels_in, dim=1, keepdims=True)

        return out
