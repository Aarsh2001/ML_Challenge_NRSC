from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists, RandomCrop
import tensorflow as tf
# import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os 
from scipy.ndimage import rotate
import torch
import albumentations as A
    

IMAGES_TO_GENERATE = 2400

images_path = './image_chips'
mask_path = './target_data'

img_aug_path = './ARG_LESS/train'
mask_aug_path = './ARG_LESS/mask'

images= []
masks = []

for im in os.listdir(images_path):
    images.append(os.path.join(images_path,im))
for msk in os.listdir(mask_path):
    masks.append(os.path.join(mask_path,msk))

aug = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # A.geometric.rotate.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
    A.HorizontalFlip(p=1),
    A.Transpose(p=1),
    A.GridDistortion(p=1),
    # A.RandomCrop(height=256,width=256),
    CropNonEmptyMaskIfExists(p=0.5,height=512,width=512),
    # A.HueSaturationValue(p=0.3),
    # A.RandomBrightnessContrast(p=0.3),
    A.augmentations.geometric.transforms.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=False, p=0.3),
    ]
    )

i=1
while i<=IMAGES_TO_GENERATE:
    number= random.randint(0,len(images)-1)
    image = images[number]
    mask = masks[number]
    print(image,mask)
    original_image =io.imread(image)
    original_mask =io.imread(mask)      
    augmented = aug(image=original_image,mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']    
    new_image_path ='%s/%s.png'%(img_aug_path,i) 
    new_mask_path ='%s/%s.png'%(mask_aug_path,i) 
    print(new_image_path)
    io.imsave(new_image_path,transformed_image)
    io.imsave(new_mask_path,transformed_mask)
    i=i+1