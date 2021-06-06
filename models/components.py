import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torch_utils.ops import upfirdn2d, fused_leaky_relu
from torch_utils.ops import conv2d_resample

# _bias_act_funcs = {
#     'linear': lambda x, b, **_: x.add_(b),
#     'lrelu': lambda x, b: F.leaky_relu(x + b),
#     'fused_lrelu': lambda x, b: fused_leaky_relu(x, b)
# }


# from: https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
# def make_kernel(k):
#     k = torch.tensor(k, dtype=torch.float32)
#     if k.ndim == 1:
#         k = k[None, :] * k[:, None]

#     k /= k.sum()
#     return k


# def get_weight(shape, gain=1, use_wscale=True, lrmul=1):
#     fan_in = np.prod(shape[1:])  # [fmaps_out, fmaps_in, kernel, kernel] or [out, in]
#     he_std = gain / np.sqrt(fan_in)  # He init

#     # Equalized learning rate and custom learning rate multiplier.
#     if use_wscale:
#         init_std = 1.0 / lrmul
#         runtime_coef = he_std * lrmul
#     else:
#         init_std = he_std / lrmul
#         runtime_coef = lrmul

#     return Parameter(torch.randn(*shape).mul_(init_std)), runtime_coef


# class Blur(nn.Module):
#     def __init__(self, kernel, pad, upsample_factor=1):
#         super().__init__()
#         kernel = make_kernel(kernel)
#         if upsample_factor > 1:
#             kernel = kernel * (upsample_factor ** 2)

#         self.register_buffer('kernel', kernel)
#         self.pad = pad

#     def forward(self, input):
#         return upfirdn2d(input, self.kernel, pad=self.pad)


# class Upsample(nn.Module):
#     def __init__(self, kernel, factor=2):
#         super().__init__()

#         self.factor = factor
#         kernel = make_kernel(kernel) * (factor ** 2)
#         self.register_buffer('kernel', kernel)

#         p = kernel.shape[0] - factor

#         pad0 = (p + 1) // 2 + factor - 1
#         pad1 = p // 2

#         self.pad = (pad0, pad1)

#     def forward(self, input):
#         return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)


# class Downsample(nn.Module):
#     def __init__(self, kernel, factor=2):
#         super().__init__()

#         self.factor = factor
#         kernel = make_kernel(kernel)
#         self.register_buffer('kernel', kernel)

#         p = kernel.shape[0] - factor

#         pad0 = (p + 1) // 2
#         pad1 = p // 2

#         self.pad = (pad0, pad1)

#     def forward(self, input):
#         return upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)


class Dense_layer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bias=True,
        bias_init=0,
        # nonlinearity='fused_lrelu',
        # negative_slope=0.2,
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
        return (f'{self.__class__.__name__}({self.out_dim}, {self.in_dim} '
                f'bias={self.use_bias}, act={self.act})')

    def forward(self, x):
        assert x.shape[1] == self.in_dim, f"unmatched shape: {x.shape[1]} v.s {self.in_dim}"
        w = self.weight.to(x.dtype) * self.weight_gain * self.lrmul
        b = self.bias * self.lrmul

        x = F.linear(x, w)

        return fused_leaky_relu(x, b)


# class ScaledLeakyReLU(nn.Module):
#     def __init__(self, negative_slope=0.2):
#         super(ScaledLeakyReLU, self).__init__()
#         self.negative_slope = negative_slope

#     def forward(self, input):
#         out = F.leaky_relu(input, negative_slope=self.negative_slope)

#         return out * math.sqrt(2)


class Conv2d_layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        mode=None,
        bias=True,
        resample_filter=[1, 3, 3, 1],
    ):
        super(Conv2d_layer, self).__init__()
        assert mode in ['up', 'down', None]

        # self.out_channels, self.in_channels, self.kernel = out_channels, in_channels, kernel
        self.mode = mode
        self.up = 2 if self.mode == "up" else 1
        self.down = 2 if self.mode == "down" else 1
        self.padding = kernel // 2

        # if self.mode == 'up':
        #     factor = 2
        #     p = (len(resample_filter) - factor) - (kernel - 1)
        #     pad0 = (p + 1) // 2 + factor - 1
        #     pad1 = p // 2 + 1
        #     self.blur = Blur(resample_filter, pad=(pad0, pad1), upsample_factor=factor)
        # elif self.mode == 'down':
        #     factor = 2
        #     p = (len(resample_filter) - factor) + (kernel - 1)
        #     pad0 = (p + 1) // 2
        #     pad1 = p // 2
        #     self.blur = Blur(resample_filter, pad=(pad0, pad1))

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.weight = Parameter(torch.randn([out_channels, in_channels, kernel, kernel]))
        self.weight_gain = 1 / np.sqrt(in_channels * kernel ** 2)
        self.bias = Parameter(torch.zeros([out_channels])) if bias else torch.empty([0])

    def forward(self, x, *args):
        weight = self.weight * self.weight_gain
        flip_weight = (self.up == 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=self.resample_filter,
                                            up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        return fused_leaky_relu(x, self.bias.to(x.device))


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
        # gain=1,
        # use_wscale=True,
        # lrmul=1
    ):
        super(Modulated_conv2d_layer, self).__init__()
        assert mode in ['up', 'down', None]

        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel = kernel
        self.mode = mode
        self.dlatent_dim = dlatent_dim
        self.demodulate = demodulate
        self.padding = kernel // 2

        self.up = 2 if self.mode == 'up' else 1
        self.down = 2 if self.mode == 'down' else 1
        # if self.mode == 'up':
        #     factor = 2
        #     p = (len(resample_filter) - factor) - (kernel - 1)
        #     pad0 = (p + 1) // 2 + factor - 1
        #     pad1 = p // 2 + 1
        #     self.blur = Blur(resample_filter, pad=(pad0, pad1),
        #                      upsample_factor=factor)
        # elif self.mode == 'down':
        #     factor = 2
        #     p = (len(resample_filter) - factor) + (kernel - 1)
        #     pad0 = (p + 1) // 2
        #     pad1 = p // 2
        #     self.blur = Blur(resample_filter, pad=(pad0, pad1))

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

        x = x.view(1, -1, h, w)
        weight = weight.reshape(-1, in_channels, *weight.shape[3:])

        # if self.mode == 'up':
        #     w = w.permute(0, 2, 1, 3, 4)
        #     w = w.reshape(-1, self.out_channels, self.kernel, self.kernel)
        #     x = F.conv_transpose2d(x, w, stride=2, padding=0, groups=b)
        # elif self.mode == 'down':
        #     x = self.blur(x)
        #     w = w.view(b * self.out_channels, -1, self.kernel, self.kernel)
        #     x = F.conv2d(x, w, stride=2, padding=0, groups=b)
        # else:
        #     w = w.view(b * self.out_channels, -1, self.kernel, self.kernel)
        #     x = F.conv2d(x, w, padding=self.padding, groups=b)

        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=self.resample_filter,
                                            up=self.up, down=self.down, padding=self.padding, groups=b, flip_weight=(self.up == 1))
        x = x.view(b, -1, *x.shape[2:])
        return x


# class NoiseInjection(nn.Module):
#     def __init__(self):
#         super(NoiseInjection, self).__init__()
#         self.noise_strength = Parameter(torch.zeros(1))

#     def forward(self, x, noise=None):
#         if noise is None:
#             # random noise
#             batch, _, height, width = x.shape
#             noise = x.new_empty(batch, 1, height, width).normal_()

#         return x + self.noise_strength * noise


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

        return fused_leaky_relu(x, self.bias)


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
            self.skip = Conv2d_layer(in_channels, out_channels, kernel=1, bias=False)

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.conv = Conv2d_layer(in_channels, in_channels, bias=False, resample_filter=resample_filter)
        self.conv_down = Conv2d_layer(in_channels, out_channels, mode='down', bias=False, resample_filter=resample_filter)

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
        fmap_base=16 << 10,          # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,              # log2 feature map reduction when doubling the resolution.
        fmap_min=1,                  # Minimum number of feature maps in any layer.
        fmap_max=512,                # Maximum number of feature maps in any layer.
        architecture='skip',         # Architecture: 'orig', 'skip', 'resnet'.
        # nonlinearity='lrelu',        # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
    ):

        super(G_synthesis_stylegan2, self).__init__()
        res_log2 = int(np.log2(img_resolution))
        self.res_log2 = res_log2
        self.resolutions = [2 ** i for i in range(2, res_log2 + 1)]

        def nf(stage):
            return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

        # act = nonlinearity
        self.architecture = architecture
        if self.architecture in ['skip', 'resnet']:
            self.trgbs = nn.ModuleList()

        # 4x4
        content_ch = 4
        face_style_dim = 112
        dlatent_dim += face_style_dim
        self.input = Parameter(torch.randn((1, nf(1), 4, 4)))
        self.style_encoder = style_encoder(res_log2, 2, 3, face_style_dim)
        self.ContentEncoder = ContentEncoder(res_log2, 2, content_ch, nf(1) // 2)
        self.bottom_layer = Layer(int(nf(1) * 1.5), int(nf(1) * 1.5), dlatent_dim, 4, resample_filter=resample_filter)
        self.trgbs.append(ToRGB(int(nf(1) * 1.5), img_channels, dlatent_dim, resample_filter=resample_filter))

        # main layers
        self.convs = nn.ModuleList()

        in_channels = int(nf(1) * 1.5)
        for res in range(3, self.res_log2 + 1):
            fmaps = int(nf(res - 1) * 1.5)
            fmaps1 = fmaps if res == self.res_log2 else nf(res - 1)
            self.convs.extend([
                Layer(in_channels, fmaps1, dlatent_dim, 2 ** res, mode='up', resample_filter=resample_filter),
                Layer(fmaps, fmaps, dlatent_dim, 2 ** res, resample_filter=resample_filter)
            ])
            if self.architecture == 'skip':
                self.trgbs.append(ToRGB(fmaps, img_channels, dlatent_dim, resample_filter=resample_filter))
            in_channels = fmaps

    def forward(self, dlatents_in, style_in=None, content_in=None, **layer_kwargs):
        tile_in = self.input.repeat(dlatents_in.shape[0], 1, 1, 1)
        content_encoding = self.ContentEncoder(content_in)
        style_encoding = self.style_encoder(style_in)
        tile_in = torch.cat([tile_in, content_encoding[-1]], dim=1)

        style_encoding = style_encoding.squeeze().unsqueeze(1).repeat(1, dlatents_in.shape[1], 1)
        dlatents_in = torch.cat([style_encoding, dlatents_in], dim=2)

        x = self.bottom_layer(tile_in, dlatents_in[:, 0], **layer_kwargs)
        if self.architecture == 'skip':
            skip = self.trgbs[0](x, dlatents_in[:, 1])

        # main layer
        for res in range(3, self.res_log2 + 1):
            x = self.convs[res * 2 - 6](x, dlatents_in[:, res * 2 - 5], **layer_kwargs)
            if res < self.res_log2:
                x = torch.cat([x, content_encoding[7 - res]], dim=1)
            x = self.convs[res * 2 - 5](x, dlatents_in[:, res * 2 - 4], **layer_kwargs)

            if self.architecture == 'skip' or res == self.res_log2:
                skip = self.trgbs[res - 2](x, dlatents_in[:, res * 2 - 3], skip=skip)
        images_out = skip

        return images_out


class G_mapping(nn.Module):
    def __init__(
        self,
        latent_size=512,         # Latent vector (Z) dimensionality.
        label_size=0,            # Label dimensionality, 0 if no labels.
        embedding_size=0,
        dlatent_dim=512,        # Disentangled latent (W) dimensionality.
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
            fmaps = dlatent_dim if layer_idx == mapping_layers - 1 else mapping_fmaps
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
