import os

import torch
import torch.nn.functional as F
from torch import nn

from .src import grid_sampler


def make_meshgrid(H, W, device='cuda:0'):
    x = torch.arange(0, W).to(device)
    y = torch.arange(0, H).to(device)

    xx, yy = torch.meshgrid([x, y])
    meshgrid = torch.stack([yy, xx], dim=0).float()

    meshgrid[0] = (2 * meshgrid[0] / (H - 1)) - 1
    meshgrid[1] = (2 * meshgrid[1] / (W - 1)) - 1

    return meshgrid


class DeformablePipe(nn.Module):
    def __init__(self, inpainter, refiner, device='cuda:0', H=256, W=256):
        super(DeformablePipe, self).__init__()
        self.inpainter = inpainter.to(device)
        self.refiner = refiner.to(device)
        self.sampler = grid_sampler.InvGridSamplerDecomposed(return_B=True, hole_fill_color=0.).to(device)
        self.device = device
        self.H = H
        self.W = W

    def save(self, folder, step):
        torch.save(self.inpainter.state_dict(), os.path.join(folder, 'inpainter{}'.format(step)))
        torch.save(self.refiner.state_dict(), os.path.join(folder, 'refiner{}'.format(step)))

    def load(self, folder, step):
        self.inpainter.load_state_dict(torch.load(os.path.join(folder, 'inpainter{}'.format(step))))
        self.refiner.load_state_dict(torch.load(os.path.join(folder, 'refiner{}'.format(step))))

    def pack_outdict(self, source_xy_textures, inpainted_xy, source_textures, pred_textures, pred_img, refined):
        outdict = {}
        outdict['source_xy_textures'] = source_xy_textures
        outdict['inpainted_xy'] = inpainted_xy
        outdict['source_textures'] = source_textures
        outdict['pred_textures'] = pred_textures
        outdict['pred_img'] = pred_img
        outdict['refined'] = refined
        return outdict

    def forward(self, inputs):
        source_img = inputs['source_img']
        source_uv = inputs['source_uv'].permute(0, 2, 3, 1)
        target_uv = inputs['target_uv'].permute(0, 2, 3, 1)

        # Generate mask of valid UV coordinates
        source_mask = torch.logical_not(torch.isnan(source_uv)).sum(dim=-1) > 0
        target_mask = torch.logical_not(torch.isnan(target_uv)).sum(dim=-1) > 0
        source_mask = source_mask.float().unsqueeze(1)
        target_mask = target_mask.float().unsqueeze(1)

        # Replace NaNs with out-of-image coordinates
        source_uv[torch.isnan(source_uv)] = -10
        target_uv[torch.isnan(target_uv)] = -10

        # Get xy textures
        N, _, H, W = source_uv.shape
        meshgrid = make_meshgrid(self.H, self.W, device=self.device)
        meshgrid = torch.stack([meshgrid] * N, dim=0)

        source_xy_textures, _ = self.sampler(meshgrid, source_uv[..., [1, 0]])

        source_fg = source_img * source_mask
        source_textures, source_holes = self.sampler(source_fg, source_uv[..., [1, 0]])
        source_holes = (source_holes[:, :1] > 1e-10).float()

        # Inpaint xy textures
        inp_in = torch.cat([source_xy_textures, source_holes[:, :1], meshgrid], dim=1)

        xy_inpainted = torch.tanh(self.inpainter(inp_in))

        # Warp source image to get RGB textures
        pred_textures = F.grid_sample(source_fg, xy_inpainted.permute(0, 2, 3, 1))

        pred_img = F.grid_sample(pred_textures, target_uv)
        pred_img = pred_img * target_mask
        target_xy = F.grid_sample(xy_inpainted, target_uv) * target_mask
        refiner_source_inp = torch.cat([source_img, source_uv.permute(0, 3, 1, 2),
                                        source_mask, meshgrid], dim=1)
        refiner_target_inp = torch.cat([pred_img, target_uv.permute(0, 3, 1, 2),
                                        target_mask, target_xy, meshgrid], dim=1)

        refined = torch.sigmoid(self.refiner(refiner_source_inp, refiner_target_inp, target_xy, target_mask))

        return self.pack_outdict(source_xy_textures, xy_inpainted, source_textures, pred_textures, pred_img, refined)

