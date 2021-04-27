import argparse
import os

import cv2
import numpy as  np

from utils.common import tti
from utils.demo import load_data, create_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--checkpoint_path', type=str, default='data/checkpoint')
    parser.add_argument('--samples_root', type=str, default='data/samples')
    parser.add_argument('--out_dir', type=str, default='data/results')
    parser.add_argument('--source_sample', type=str)
    parser.add_argument('--target_sample', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = args.device

    data_dict = load_data(args.samples_root, args.source_sample, args.target_sample, device=device)

    pipeline = create_pipeline(args.checkpoint_path, device=device)
    output_dict = pipeline(data_dict)

    pred_img = (tti(output_dict['refined']) * 255).astype(np.uint8)

    pair_str = args.source_sample + '_to_' + args.target_sample
    out_path = os.path.join(args.out_dir, pair_str + '.png')
    os.makedirs(args.out_dir, exist_ok=True)
    
    print('Writing the .png result to:', out_path)
    cv2.imwrite(out_path, pred_img[..., ::-1])
