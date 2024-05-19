"""
Author: Dr. Sreenivas Bhattiprolu

Semantic segmentation + watershed --> Instance segmentation

Prediction for semantic segmentation (Unet) of mitochondria
Uses model trained on the standard Unet framework with no tricks!

Dataset info: Electron microscopy (EM) dataset from
https://www.epfl.ch/labs/cvlab/data/data-em/

Patches of 256x256 from images and labels 
have been extracted (via separate program) and saved to disk. 

This code performs segmentation of 256x256 images followed by watershed
based separation of objects. Object properties will also be calculated.

"""
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io


def get_instances(img, param=0.05):
    img_grey = img[:,:,0]
    
    _, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 6)
    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, param*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown==255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0,0,255]  
    img2 = color.label2rgb(markers, bg_label=0)
    # cv2.imshow('Overlay on original image', img)
    # cv2.imshow('Colored Grains', img2)
    # cv2.waitKey(0)
    props = measure.regionprops_table(markers, intensity_image=img_grey, 
                                properties=['label',
                                            'area', 'equivalent_diameter',
                                            'mean_intensity', 'solidity'])
    
    all_colors = np.unique(img2.reshape(-1, img2.shape[2]), axis=0)
    colours, counts = np.unique(img2.reshape(-1,3), axis=0, return_counts=1)
    import pandas as pd
    df = pd.DataFrame(props)
    df = df[df.mean_intensity > 0.5]

    segments_in_img = []

    ignore_colors = [[1., 0., 0.], [0., 0., 1.]]

    for seg_color, seg_area in zip(colours, counts):
        if any(np.array_equal(seg_color, ignore_color) for ignore_color in ignore_colors):
            continue
        if seg_area < 500:
            continue
        seg_mask = (img2 == seg_color).all(-1)
        segments_in_img.append(seg_mask)

    return segments_in_img


