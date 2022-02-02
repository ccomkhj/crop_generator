import cv2
import os
import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from crop_gen_auto import group

from loguru import logger

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Crop image generator")

    parser.add_argument("--mask",
                        default='input/train',
                        help="Location of mask directory.")

    parser.add_argument("--output",
                        default='output/train',
                        help="Location of output directory to save cropped images.")


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    masks_name = sorted(glob.glob(os.path.join(args.mask, '*')))
    out_dir = os.path.join(args.output)

    if not os.path.exists(out_dir): # if no directory, then make it.
        logger.info(f'new directory: {out_dir} is generated.')
        os.makedirs(out_dir)

    for mask_name in masks_name:

        logger.info('This image is being processed : ', mask_name)
        cropped_masks, rois = group(mask_name, sensitivity= 0.0001, intact=True)
        base_name = os.path.basename(mask_name).split('.')

        for order, cropped_mask in enumerate(cropped_masks):
            crop_name = os.path.join(out_dir, base_name[0] + f'_{str(order)}.' + base_name[1])
            cv2.imwrite(crop_name, cropped_mask)



       

'''
- TIP
There will be some images which is cropped. It happens when an image is located on the border.
If you don't need it, keep intact=True
'''

        
