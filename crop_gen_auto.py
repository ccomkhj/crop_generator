import cv2
import numpy as np
import os
import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from loguru import logger

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Crop image generator")

    parser.add_argument("--image",
                        default='input/image',
                        help="Location of input image directory to be cropped.")

    parser.add_argument("--mask",
                        default='input/mask',
                        help="Location of mask directory.")

    parser.add_argument("--output",
                        default='output',
                        help="Location of output directory to save cropped images.")

    parser.add_argument("--type",
                        default='kopf_salad',
                        help="Types of the cropped image.")


    args = parser.parse_args()
    return args

def group(mask, sensitivity, debug=False, intact=True):
    '''
    group independent objects.

    Input
    mask: location of segmentation mask (after segmentation process)
    sensitivity : this ratio is the parameter to decide, based on the object size (recommend: 0.0001 ~ 0.01)
    intact: keep it True, if you want to ignore object at the border

    Output
    cropped_masks: a list of cropped object masks
    roi: region of interest for each object mask

    '''

    img = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY)
    img = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE, kernel=(3,3))

    sketch = img.copy()
    num_groups, group_mask, bboxes, centers  = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
    logger.info(f'The number of {num_groups-1} objects are detected.')
    # bboxes: left most x, top most y, horizontal size, vertical size, total area 
    

    
    MIN_AREA = img.shape[0] * img.shape[1] * sensitivity 

    cropped_masks = []
    rois = []

    for i, (bbox, center) in enumerate(zip(bboxes, centers)):
        tx, ty, hori, verti, area = bbox
        if area < MIN_AREA or i == 0:
            # skip too small or the whole image
            continue
        cx, cy = center
        roi = ty, ty+verti, tx, tx+hori

        cropped = img[roi[0]:roi[1], roi[2]:roi[3]]

        if cv2.connectedComponents(cropped)[0] != 2: # if there is more than one object in the cropped mask,
            continue

        if intact and any(roi) == 0 | img.shape[0]-1 | img.shape[1]-1: # if a cropped image is located on the image border,
            continue

        cropped_masks.append(cropped)
        rois.append(roi)

        if debug:
            sketch = cv2.rectangle(sketch, pt1= (tx, ty), pt2= (tx+hori, ty+verti), color= 1, thickness= 2)
            f, axarr = plt.subplots(2)
            axarr[0].imshow(sketch)
            axarr[1].imshow(cropped)
            plt.show()
    logger.info(f'The number of {len(cropped_masks)} objects are saved.')

    return cropped_masks, rois

def crop_img(rois, img_name):
    name_parts = os.path.basename(img_name).split('.')

    for order, roi in enumerate(rois):
        # crop relevant original image area.
        crop_name = os.path.join(out_dir, name_parts[0] + f'_{str(order)}.' + name_parts[1])
        cropped_img = cv2.imread(img_name)[roi[0]:roi[1], roi[2]:roi[3]]
        cv2.imwrite(crop_name, cropped_img)
        logger.info(f'cropped image: {crop_name} is generated.')

        
        
if __name__ == '__main__':
    args = parse_args()
    img_names = sorted(glob.glob(os.path.join(args.image, '*')))
    masks = sorted(glob.glob(os.path.join(args.mask, '*')))
    out_dir = os.path.join(args.output, args.type)

    if not os.path.exists(out_dir): # if no directory, then make it.
        logger.info(f'new directory: {out_dir} is generated.')
        os.makedirs(out_dir)

    for img_name, mask in zip(img_names, masks):

        logger.info('This image is being processed : ', mask)
        _, rois = group(mask, intact=True)
        crop_img(rois, img_name)


       

'''
- TIP
There will be some images which is cropped. It happens when an image is located on the border.
If you don't need it, keep intact=True
'''

        
