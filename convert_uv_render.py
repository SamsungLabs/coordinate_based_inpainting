import argparse
import os

import cv2
import numpy as np

from utils import smpltex
from utils.demo import iuv2smpluv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iuv_dir', type=str, default='data/samples/source_iuv/')
    parser.add_argument('--out_dir', type=str, default='data/samples/source_uv/')
    parser.add_argument('--sample_id', type=str)
    parser.add_argument('--mapping_file', type=str, default='data/smpltexmap.npy')
    args = parser.parse_args()

    transformer = smpltex.TexTransformer(args.mapping_file)
    
    iuv_path = os.path.join(args.iuv_dir, args.sample_id+'_IUV.png')
    out_path = os.path.join(args.out_dir, args.sample_id+'.npy')

    iuv_png = cv2.imread(iuv_path)[..., ::-1]
    uv_smpl = iuv2smpluv(iuv_png, transformer)

    print('Saved the converted UV to: ', out_path)
    np.save(out_path, uv_smpl)
