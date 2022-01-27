import cv2
import numpy as np
import os
import argparse
from loguru import logger

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
    global croppedImage

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False  # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2:  # when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            croppedImage = roi
            cv2.imshow("Cropped", roi)

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Crop image generator")

    parser.add_argument("--input",
                        default='input',
                        help="Location of input directory to be cropped.")

    parser.add_argument("--output",
                        default='output',
                        help="Location of output directory to save cropped images.")

    parser.add_argument("--type",
                        default='basil',
                        help="Types of the cropped image.")
    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()        
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    dir = args.input
    out_dir = os.path.join(args.output, args.type)

    logger.info('Input is located in '+ dir)
    

    if not os.path.exists(out_dir): # if no directory, then make it.
        logger.info('new directory is generated.')
        os.makedirs(out_dir)

    logger.info('Cropped images will be saved in '+out_dir)

    global image, oriImage

    for file in os.listdir(dir):

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)

        fullpath = os.path.join(dir, file)

        image = cv2.imread(fullpath)
        oriImage = image.copy()

        name_parts = os.path.basename(file).split('.')

        order = 0

        key = cv2.waitKey(1) & 0xFF
        while True:
            i = image.copy()

            if not cropping:
                cv2.imshow("image", image)

            elif cropping:
                cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("image", i)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'): # Quit croping in the current image
                cv2.destroyAllWindows()

                logger.info('Try the next cropping.')

                cv2.namedWindow("image")
                cv2.setMouseCallback("image", mouse_crop)
                pass

            if key == ord('x'): # save and eXist
                crop_name = os.path.join(out_dir, name_parts[0]+'_'+str(order)+'.'+name_parts[1])
                cv2.imwrite(crop_name , croppedImage)
                cv2.destroyAllWindows()

                logger.info(crop_name+ ' is saved. Move to the next image.')
                break

            if key == ord('c'): # Continue cropping in the current image
                crop_name = os.path.join(out_dir, name_parts[0]+'_'+str(order)+'.'+name_parts[1])
                cv2.imwrite(crop_name , croppedImage)
                cv2.destroyAllWindows()

                logger.info(crop_name+ ' is saved. More cropping is working on the same image.')
                order += 1    

                cv2.namedWindow("image")
                cv2.setMouseCallback("image", mouse_crop)
                pass

            if key == ord('q'):
                cv2.destroyAllWindows()
                logger.info('Move to the next image.')
                break

    # close all open windows
    cv2.destroyAllWindows()