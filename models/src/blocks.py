import torch.nn as nn
import torch.nn.functional as F
from .spectral_norm import SpectralNorm


class _conv_block(nn.Module):
    def __init__(self, in_features, out_features, filter_size=3, stride=1, dilation=1,
                 padding_mode='reflect', sn=False, act_fun=nn.LeakyReLU, normalization=nn.BatchNorm2d):
        super(_conv_block, self).__init__()
        self.pad_mode = padding_mode
        self.filter_size = filter_size
        self.dilation = dilation

        self.conv = nn.Conv2d(in_features, out_features, filter_size, stride=stride, dilation=dilation)
        if sn:
            self.conv = SpectralNorm(self.conv)

        if normalization is not None:
            self.norm = normalization(out_features)
        else:
            self.norm = None
        self.act_f = act_fun()

    def forward(self, x):
        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)
        n_pad_by_sides = (n_pad_pxl, n_pad_pxl, n_pad_pxl, n_pad_pxl)
        output = self.conv(F.pad(x, n_pad_by_sides, mode=self.pad_mode))
        if self.norm is not None:
            output = self.norm(output)
        output = self.act_f(output)
        return output


class _gated_conv_block(nn.Module):
    def __init__(self, in_features, out_features, filter_size=3, stride=1, dilation=1,
                 padding_mode='reflect', sn=False, normalization=nn.BatchNorm2d, act_fun=nn.ELU):
        super(_gated_conv_block, self).__init__()
        self.pad_mode = padding_mode
        self.filter_size = filter_size
        self.stride = stride
        self.dilation = dilation

        self.conv_f = nn.Conv2d(in_features, out_features, filter_size, stride=stride, dilation=dilation)
        if sn:
            self.conv_f = SpectralNorm(self.conv_f)
        if normalization is not None:
            self.norm = normalization(out_features)
        else:
            self.norm = None
        self.act_f = act_fun()

        self.conv_m = nn.Conv2d(in_features, out_features, filter_size, stride=stride, dilation=dilation)
        if sn:
            self.conv_m = SpectralNorm(self.conv_m)
        self.act_m = nn.Sigmoid()

    def forward(self, x):
        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)
        n_pad_by_sides = (n_pad_pxl, n_pad_pxl, n_pad_pxl, n_pad_pxl)
        x_padded = F.pad(x, n_pad_by_sides, mode=self.pad_mode)

        features = self.act_f(self.conv_f(x_padded))
        mask = self.act_m(self.conv_m(x_padded))
        output = features * mask
        if self.norm is not None:
            output = self.norm(output)

        return output


class _final_conv_block(nn.Module):
    def __init__(self, in_features, out_features, filter_size=3, stride=1, padding_mode='reflect', sn=False):
        super(_final_conv_block, self).__init__()
        self.pad_mode = padding_mode
        self.filter_size = filter_size

        self.conv1 = nn.Conv2d(in_features, out_features, filter_size, stride=stride)
        if sn:
            self.conv1 = SpectralNorm(self.conv1)

    def forward(self, x):
        n_pad_pxl = self.filter_size // 2
        n_pad_by_sides = (n_pad_pxl, n_pad_pxl, n_pad_pxl, n_pad_pxl)

        output = self.conv1(F.pad(x, n_pad_by_sides, mode=self.pad_mode))

        return output
