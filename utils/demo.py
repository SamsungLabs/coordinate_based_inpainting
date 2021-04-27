import os

import cv2
import numpy as np
import torch

from models import pipelines
from models.src import dp
from utils.common import itt


def iuv2smpluv(iuv, transformer):
    inds_png = iuv[..., 2]
    uv_png = iuv[..., :2]

    uv_transformed = transformer.GetGlobalUV(inds_png, uv_png)
    return uv_transformed


def load_data(samples_root, source_sample, target_sample, device='cuda:0'):
    source_uv_path = os.path.join(samples_root, 'source_uv', source_sample + '.npy')
    source_img_path = os.path.join(samples_root, 'source_img', source_sample + '.jpg')
    target_uv_path = os.path.join(samples_root, 'target_uv', target_sample + '.npy')

    source_uv = np.load(source_uv_path)
    source_img = cv2.imread(str(source_img_path))[..., [2, 1, 0]] / 255
    target_uv = np.load(target_uv_path)

    [source_uv, source_img, target_uv] = [itt(x).unsqueeze(0).to(device) for x in
                                          [source_uv, source_img, target_uv]]

    return dict(source_uv=source_uv,
                source_img=source_img,
                target_uv=target_uv)


def create_pipeline(checkpoint_path='data/checkpoint', device='cuda:0'):
    inpainter_file = os.path.join(checkpoint_path, 'inpainter.pth')
    refiner_file = os.path.join(checkpoint_path, 'refiner.pth')

    inpainter = dp.GatedHourglass(32, 5, 2).to(device)
    refiner = dp.ResHourglassDeformableSkip(8, 10, 3, ngf=256).to(device)

    inpainter.load_state_dict(torch.load(inpainter_file))
    refiner.load_state_dict(torch.load(refiner_file))

    pipeline = pipelines.DeformablePipe(inpainter, refiner).eval()
    return pipeline
