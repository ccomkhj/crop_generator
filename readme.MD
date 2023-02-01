# Cropping Image Generator

### This is useful to get multiple (or single) cropped images from input image.


## crop_gen_manual.py

### This is helpful tool to manually crop images.

### 1. Run the code
```bash
python crop_gen_manual.py --input {input directory} --output {output directory} --type {type of cropped image} \
ex) python crop_gen_manual.py --input input --output output --type basil 
```

### 2.Drag from top left to bottom right. 
![plot](./demo/crop.jpg)

### 3. Check if the cropped image is okay. 
3-1. Want to save? and more cropping in the same image? --> press **c**  \
3-2. Don't want to save? and recropping in the same image? --> press **r**  \
3-3. Don't want to save? move to the next image? --> press **q** 

## crop_gen_auto.py

### This is helpful tool to automatically crop images, when you have mask.
### It is effective after image segmentation.

## crop_gen_mask.py

### This is helpful tool to get individual object from objects.
