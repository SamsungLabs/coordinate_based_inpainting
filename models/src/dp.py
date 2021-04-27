import torch
import torch.nn as nn
import torch.nn.functional as F
from . import blocks


class GatedHourglass(nn.Module):
    def __init__(self, ngf, in_channels, out_channels):
        super(GatedHourglass, self).__init__()
        self.ngf = ngf

        main_block = blocks._gated_conv_block
        normalization = nn.BatchNorm2d
        sn = False

        self.gate1 = main_block(in_channels, ngf, 5, 1, normalization=normalization, sn=sn)
        self.gate2 = main_block(ngf, ngf * 2, 3, stride=2, normalization=normalization, sn=sn)
        self.gate3 = main_block(ngf * 2, ngf * 2, 3, 1, normalization=normalization, sn=sn)
        self.gate4 = main_block(ngf * 2, ngf * 4, 3, 2, normalization=normalization, sn=sn)
        self.gate5 = main_block(ngf * 4, ngf * 4, 3, 1, normalization=normalization, sn=sn)
        self.gate6 = main_block(ngf * 4, ngf * 4, 3, 1, normalization=normalization, sn=sn)
        self.gate7 = main_block(ngf * 4, ngf * 4, 3, 1, dilation=2, normalization=normalization, sn=sn)
        self.gate8 = main_block(ngf * 4, ngf * 4, 3, 1, dilation=4, normalization=normalization, sn=sn)
        self.gate9 = main_block(ngf * 4, ngf * 4, 3, 1, dilation=8, normalization=normalization, sn=sn)
        self.gate10 = main_block(ngf * 4, ngf * 4, 3, 1, dilation=8, normalization=normalization, sn=sn)
        self.gate11 = main_block(ngf * 4, ngf * 4, 3, 1, normalization=normalization, sn=sn)
        self.gate12 = main_block(ngf * 4, ngf * 4, 3, 1, normalization=normalization, sn=sn)
        self.upsample13 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.gate13 = main_block(ngf * 4, ngf * 2, 3, 1, normalization=normalization, sn=sn)
        self.upsample14 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.gate14 = main_block(ngf * 2, ngf, 3, 1, normalization=normalization, sn=sn)
        self.gate15 = main_block(ngf, ngf // 2, 3, 1, normalization=normalization, sn=sn)
        self.final = blocks._final_conv_block(ngf // 2, out_channels, sn=sn)

        modlist = []
        modlist.append(self.gate1)
        modlist.append(self.gate2)
        modlist.append(self.gate3)
        modlist.append(self.gate4)
        modlist.append(self.gate5)
        modlist.append(self.gate6)
        modlist.append(self.gate7)
        modlist.append(self.gate8)
        modlist.append(self.gate9)
        modlist.append(self.gate10)
        modlist.append(self.gate11)
        modlist.append(self.gate12)
        modlist.append(self.upsample13)
        modlist.append(self.gate13)
        modlist.append(self.upsample14)
        modlist.append(self.gate14)
        modlist.append(self.gate15)

        self.sequence = nn.Sequential(*modlist)

    def forward(self, x):
        out = self.sequence(x)
        out = self.final(out)

        return out


class ResHourglassDeformableSkip(nn.Module):
    def __init__(self, in_source, in_target, out_features, ngf=256):
        super(ResHourglassDeformableSkip, self).__init__()

        self.ngf = ngf
        main_block = blocks._conv_block
        normalization = nn.InstanceNorm2d
        sn = True

        self.source_enc1 = main_block(in_source, ngf, 5, 1, normalization=normalization, sn=sn)
        self.source_enc2 = main_block(ngf, ngf, 3, 2, normalization=normalization, sn=sn)
        self.source_enc3 = main_block(ngf, ngf, 3, 2, normalization=normalization, sn=sn)
        self.target_enc1 = main_block(in_target, ngf, 5, 1, normalization=normalization, sn=sn)
        self.target_enc2 = main_block(ngf, ngf, 3, 2, normalization=normalization, sn=sn)
        self.target_enc3 = main_block(ngf, ngf, 3, 2, normalization=normalization, sn=sn)
        self.res1 = main_block(ngf, ngf, 3, 1, normalization=normalization, sn=sn)
        self.res2 = main_block(ngf, ngf, 3, 1, normalization=normalization, sn=sn)
        self.res3 = main_block(ngf, ngf, 3, 1, normalization=normalization, sn=sn)
        self.res4 = main_block(ngf, ngf, 3, 1, normalization=normalization, sn=sn)
        self.res5 = main_block(ngf, ngf, 3, 1, normalization=normalization, sn=sn)
        self.res6 = main_block(ngf, ngf, 3, 1, normalization=normalization, sn=sn)
        self.dec1 = main_block(ngf * 3, ngf, 3, 1, normalization=normalization, sn=sn)
        self.dec2 = main_block(ngf * 3, ngf, 3, 1, normalization=normalization, sn=sn)
        self.dec3 = main_block(ngf * 3, ngf, 3, 1, normalization=normalization, sn=sn)
        self.final = blocks._final_conv_block(ngf, out_features, 3, 1, sn=sn)

    def make_meshgrid(self, H, W, device):
        x = torch.arange(0, W).to(device)
        y = torch.arange(0, H).to(device)

        xx, yy = torch.meshgrid([x, y])
        meshgrid = torch.stack([yy, xx], dim=0).float()

        meshgrid[0] = (2 * meshgrid[0] / (H - 1)) - 1
        meshgrid[1] = (2 * meshgrid[1] / (W - 1)) - 1

        return meshgrid

    def forward(self, source, target, warpfield, target_mask):
        N, _, H, W = warpfield.shape
        device = warpfield.device

        target_mask_ds1 = (F.interpolate(target_mask, (H // 2, W // 2), mode='nearest') > 0).float()
        target_mask_ds2 = (F.interpolate(target_mask_ds1, (H // 4, W // 4), mode='nearest') > 0).float()

        meshgrid = self.make_meshgrid(H, W, device)
        meshgrid_ds1 = self.make_meshgrid(H // 2, W // 2, device)
        meshgrid_ds2 = self.make_meshgrid(H // 4, W // 4, device)

        warpfield_ds1 = F.interpolate(warpfield, (H // 2, W // 2), mode='bilinear')
        warpfield_ds2 = F.interpolate(warpfield_ds1, (H // 4, W // 4), mode='bilinear')
        warpfield = warpfield * target_mask + meshgrid * (1. - target_mask)
        warpfield_ds1 = warpfield_ds1 * target_mask_ds1 + meshgrid_ds1 * (1. - target_mask_ds1)
        warpfield_ds2 = warpfield_ds2 * target_mask_ds2 + meshgrid_ds2 * (1. - target_mask_ds2)

        source_enc1_out = self.source_enc1(source)
        source_enc2_out = self.source_enc2(source_enc1_out)
        source_enc3_out = self.source_enc3(source_enc2_out)

        source_enc1_warped = F.grid_sample(source_enc1_out, warpfield.permute(0, 2, 3, 1))
        source_enc2_warped = F.grid_sample(source_enc2_out, warpfield_ds1.permute(0, 2, 3, 1))
        source_enc3_warped = F.grid_sample(source_enc3_out, warpfield_ds2.permute(0, 2, 3, 1))

        target_enc1_out = self.target_enc1(target)
        target_enc2_out = self.target_enc2(target_enc1_out)
        target_enc3_out = self.target_enc3(target_enc2_out)

        res1_out = self.res1(target_enc3_out)
        res2_out = self.res2(target_enc3_out + res1_out)
        res3_out = self.res3(res1_out + res2_out)
        res4_out = self.res4(res2_out + res3_out)
        res5_out = self.res5(res3_out + res4_out)
        res6_out = self.res6(res4_out + res5_out)

        dec1_in = torch.cat([res5_out + res6_out, target_enc3_out, source_enc3_warped], dim=1)
        dec1_out = self.dec1(dec1_in)
        dec1_up = F.interpolate(dec1_out, scale_factor=2, mode='bilinear')
        dec2_in = torch.cat([dec1_up, target_enc2_out, source_enc2_warped], dim=1)
        dec2_out = self.dec2(dec2_in)
        dec2_up = F.interpolate(dec2_out, scale_factor=2, mode='bilinear')
        dec3_in = torch.cat([dec2_up, target_enc1_out, source_enc1_warped], dim=1)
        dec3_out = self.dec3(dec3_in)
        out = self.final(dec3_out)
        return out
