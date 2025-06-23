#https://github.com/chaiNNer-org/spandrel/blob/4976eed57ac60bc939fb349f65783bf50fbb40e5/libs/spandrel/spandrel/util/__init__.py#
#https://github.com/muslll/neosr/blob/master/neosr/archs/span_arch.py



from collections import OrderedDict

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from torch import nn
import sys
training = False
upscale=4

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1
        self.training = training

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        )

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)

        if self.training is False:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False
            self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat


    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out

class SPAB(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.c3_r(out2_act))

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att

class span(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 feature_channels=48,
                 upscale=upscale,
                 bias=True,
                 norm=False,
                 img_range=1.0,
                 rgb_mean=(0.5, 0.5, 0.5)
                 ):
        super(span, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.no_norm: torch.Tensor | None
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None

        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB(feature_channels, bias=bias)
        self.block_2 = SPAB(feature_channels, bias=bias)
        self.block_3 = SPAB(feature_channels, bias=bias)
        self.block_4 = SPAB(feature_channels, bias=bias)
        self.block_5 = SPAB(feature_channels, bias=bias)
        self.block_6 = SPAB(feature_channels, bias=bias)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

    @property
    def is_norm(self):
        return self.no_norm is None

    def forward(self, x):
        if self.is_norm:
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        out_b1, _, att1 = self.block_1(out_feature)
        out_b2, _, att2 = self.block_2(out_b1)
        out_b3, _, att3 = self.block_3(out_b2)

        out_b4, _, att4 = self.block_4(out_b3)
        out_b5, _, att5 = self.block_5(out_b4)
        out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)

        return output
import math

# code stolen from spandrel
def get_scale_and_output_channels(x: int, input_channels: int) -> tuple[int, int]:
    """
    Returns a scale and number of output channels such that `scale**2 * out_nc = x`.

    This is commonly used for pixelshuffel layers.
    """
    # Unfortunately, we do not have enough information to determine both the scale and
    # number output channels correctly *in general*. However, we can make some
    # assumptions to make it good enough.
    #
    # What we know:
    # - x = scale * scale * output_channels
    # - output_channels is likely equal to input_channels
    # - output_channels and input_channels is likely 1, 3, or 4
    # - scale is likely 1, 2, 4, or 8

    def is_square(n: int) -> bool:
        return math.sqrt(n) == int(math.sqrt(n))

    # just try out a few candidates and see which ones fulfill the requirements
    candidates = [input_channels, 3, 4, 1]
    for c in candidates:
        if x % c == 0 and is_square(x // c):
            return int(math.sqrt(x // c)), c

    raise AssertionError(
        f"Expected output channels to be either 1, 3, or 4."
        f" Could not find a pair (scale, out_nc) such that `scale**2 * out_nc = {x}`"
    )

class CSELayer(nn.Module):
    def __init__(self, num_channels: int = 48, reduction_ratio: int = 2):
        super(CSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.squeezing = nn.Sequential(
            nn.Conv2d(num_channels, num_channels_reduced, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_channels_reduced, num_channels, 1, 1),
            nn.Hardsigmoid(True),
        )

    def forward(self, input_tensor):
        squeeze_tensor = torch.mean(input_tensor, dim=[2, 3], keepdim=True)
        output_tensor = input_tensor * self.squeezing(squeeze_tensor)
        return output_tensor


# channels_first
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.offset = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-1.0 / 2))
        x_normed = x / (rms_x + self.eps)
        return self.scale[..., None, None] * x_normed + self.offset[..., None, None]


class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain: int = 2, s: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )
        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()  # type: ignore
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2  # noqa: N806
        W_pixels_to_pad = (target_kernel_size - 1) // 2  # noqa: N806
        sk_w = F.pad(
            sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat  # type: ignore

    def forward(self, x):  # noqa: ANN201, ANN001
        x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
        out = self.conv(x_pad) + self.sk(x)
        return out


class SeqConv3x3(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.mid_planes = int(out_planes * depth_multiplier)
        conv0 = torch.nn.Conv2d(
            self.inp_planes, self.mid_planes, kernel_size=1, padding=0
        )
        self.k0 = conv0.weight
        self.b0 = conv0.bias

        conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
        self.k1 = conv1.weight
        self.b1 = conv1.bias

    def forward(self, x):
        # conv-1x1
        y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
        # explicitly padding with bias
        y0 = F.pad(y0, (1, 1, 1, 1), "constant", 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        # conv-3x3
        return F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None
        # re-param conv kernel
        RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
        # re-param conv bias
        RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(
            1, -1, 1, 1
        )
        RB = (
            F.conv2d(input=RB, weight=self.k1).view(
                -1,
            )
            + self.b1
        )
        return RK, RB


class RepConv(nn.Module):
    def __init__(self, in_dim=3, out_dim=32):
        super().__init__()
        self.conv1 = SeqConv3x3(in_dim, out_dim, 2)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv3 = Conv3XC(in_dim, out_dim)
        self.conv_3x3_rep = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.alpha = nn.Parameter(torch.randn(3), requires_grad=True)
        self.forward_module = self.train_forward

        nn.init.constant_(self.alpha, 1.0)

    def fuse(self):
        conv1_w, conv1_b = self.conv1.rep_params()
        conv2_w, conv2_b = self.conv2.weight, self.conv2.bias
        self.conv3.update_params()
        conv3_w, conv3_b = self.conv3.eval_conv.weight, self.conv3.eval_conv.bias
        device = self.conv_3x3_rep.weight.device
        sum_weight = (
            self.alpha[0] * conv1_w + self.alpha[1] * conv2_w + self.alpha[2] * conv3_w
        ).to(device)
        sum_bias = (
            self.alpha[0] * conv1_b + self.alpha[1] * conv2_b + self.alpha[2] * conv3_b
        ).to(device)
        self.conv_3x3_rep.weight = nn.Parameter(sum_weight)
        self.conv_3x3_rep.bias = nn.Parameter(sum_bias)

    def train_forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.alpha[0] * x1 + self.alpha[1] * x2 + self.alpha[2] * x3

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.fuse()
        return self

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.conv_3x3_rep(x)


class OmniShift(nn.Module):
    def __init__(self, dim: int = 48) -> None:
        super().__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=True
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=True,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=True,
        )
        self.alpha1 = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        self.alpha3 = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        self.alpha4 = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=True,
        )

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        out = (
            self.alpha1 * x
            + self.alpha2 * out1x1
            + self.alpha3 * out3x3
            + self.alpha4 * out5x5
        )
        return out

    def reparam_5x5(self) -> None:
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))

        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))
        combined_weight = (
            self.alpha1.transpose(0, 1) * identity_weight
            + self.alpha2.transpose(0, 1) * padded_weight_1x1
            + self.alpha3.transpose(0, 1) * padded_weight_3x3
            + self.alpha4.transpose(0, 1) * self.conv5x5.weight
        )

        combined_bias = (
            self.alpha2.squeeze() * self.conv1x1.bias
            + self.alpha3.squeeze() * self.conv3x3.bias
            + self.alpha4.squeeze() * self.conv5x5.bias
        )

        device = self.conv5x5_reparam.weight.device

        combined_weight = combined_weight.to(device)
        combined_bias = combined_bias.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)
        self.conv5x5_reparam.bias = nn.Parameter(combined_bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.reparam_5x5()

    def forward(self, x):
        if self.training:
            out = self.forward_train(x)
        else:
            out = self.conv5x5_reparam(x)
        return out


class ParPixelUnshuffle(nn.Module):
    def __init__(self, in_dim, out_dim, down):
        super().__init__()
        self.pu = nn.PixelUnshuffle(down)
        self.poll = nn.Sequential(
            nn.MaxPool2d(kernel_size=down, stride=down), RepConv(in_dim, out_dim)
        )

    def forward(self, x):
        return self.pu(x) + self.poll(x)


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 8 / 3,
        conv_ratio: float = 1.0,
        dccm: bool = True,
        se: bool = False,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = RepConv(dim, hidden * 2)
        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.conv = nn.Sequential(
            ParPixelUnshuffle(dim, dim * 4, 2),
            OmniShift(dim * 4),
            CSELayer(dim * 4) if se else nn.Identity(),
            nn.PixelShuffle(2),
        )  # InceptionDWConv2d(dim*4)
        self.fc2 = RepConv(hidden, dim) if dccm else nn.Conv2d(hidden, dim, 1, 1)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return x + shortcut

class RTMoSR(nn.Module):
    hyperparameters = {}
    def __init__(
        self,
        *,
        scale: int = 1,
        dim: int = 32,
        ffn_expansion: float = 2,
        n_blocks: int = 2,
        unshuffle_mod: bool = False,
        dccm: bool = True,
        se: bool = True,
    ):
        super().__init__()
        self.scale = scale
        unshuffle = 0
        if scale < 4 and unshuffle_mod:
            if scale == 3:
                raise ValueError("Unshuffle_mod does not support 3x")
            unshuffle = 4 // scale
            scale = 4
        self.pad = unshuffle if unshuffle > 0 else 1
        self.pad *= 2
        self.to_feat = (
            RepConv(3, dim)
            if not unshuffle
            else nn.Sequential(
                nn.PixelUnshuffle(unshuffle), RepConv(3 * unshuffle * unshuffle, dim)
            )
        )
        self.body = nn.Sequential(
            *[
                GatedCNNBlock(dim, ffn_expansion, dccm=dccm, se=se)
                for _ in range(n_blocks)
            ]
        )
        self.to_img = nn.Sequential(
            RepConv(dim, 3 * scale**2),
            nn.PixelShuffle(scale),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_img_size(self, x, resolution: tuple[int, int]):
        scaled_size = self.pad
        mod_pad_h = (scaled_size - resolution[0] % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - resolution[1] % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.check_img_size(x, (h, w))
        out = self.to_feat(out)
        out = self.body(out)
        return self.to_img(out)[
            :, :, : h * self.scale, : w * self.scale
        ] + F.interpolate(x, scale_factor=self.scale)



def RTMoSR_L(**kwargs):
    return RTMoSR(unshuffle_mod=True, **kwargs)


def RTMoSR_UL(**kwargs):
    return RTMoSR(ffn_expansion=1.5, dccm=False, unshuffle_mod=True, **kwargs)

model = RTMoSR_L(scale=1)
model.eval()
from safetensors.torch import load_file
state_dict = load_file("1xDeH264_RTMoSR_Unshuffle.safetensors")
model.load_state_dict(state_dict, strict=True)

with torch.inference_mode():
    mod = torch.jit.trace(model, torch.rand(1, 3, 256, 256))
    mod.save('1xDeH264_RTMoSR.pt')
    """torch.onnx.export(
        model,
        torch.rand(1, 3, 256, 256),
        "1xDeH264_RTMoSR.onnx",
        verbose=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        #dynamic_axes={
        #        "input": {0: "batch_size", 2: "width", 3: "height"},
        #        "output": {0: "batch_size", 2: "width", 3: "height"},
        #    }
    )"""
import cv2
image = cv2.imread("images/in0.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))
image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
from spandrel import ModelLoader
model = ModelLoader().load_from_file("1xDeH264_RTMoSR_Unshuffle.pth")
with torch.no_grad():
    output = model(image)
    output = output.squeeze(0).permute(1, 2, 0).cpu()
output = (output * 255.0).clamp(0, 255).byte().numpy()
output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
cv2.imwrite("output.png", output)
