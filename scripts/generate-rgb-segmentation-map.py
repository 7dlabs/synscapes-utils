#! /usr/bin/env python3

import argparse
import math
import os
import sys

import numpy as np
import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from helpers import *

# Arguments ---

parser = argparse.ArgumentParser(
    description='Displays bounding boxes over SynScapes images.')

parser.add_argument('path',
                    help='Path to SynScapes root directory.')

args = parser.parse_args()

# Constants ---

class_id_to_str = {
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    31: 'train',
    32: 'motorcyclist',
    33: 'bicyclist'
}

colors = [
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (111, 74,  0),
    ( 81,  0, 81),
    (128, 64,128),
    (244, 35,232),
    (250,170,160),
    (230,150,140),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (180,165,180),
    (150,100,100),
    (150,120, 90),
    (153,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0,  0, 90),
    (  0,  0,110),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32)
]

colormap = matplotlib.colors.ListedColormap(
    [(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0) for c in colors])

# Functions ---


# Script ---

root = os.path.abspath(args.path)

# Ensure root exists
if not os.path.exists(root):
    print('Invalid path:', root)
    sys.exit(1)

img_dir, meta_dir = [os.path.join(root, x) for x in ['img', 'meta']]

# Ensure integrity
for d in [img_dir, meta_dir]:
    if not os.path.exists(d):
        print('Missing directory:', d)
        sys.exit(1)

if not os.path.exists(os.path.join(root, 'img', 'class_rgb')):
    os.mkdir(os.path.join(root, 'img', 'class_rgb'))

indices = sorted([int(f.split('.')[0]) for f in os.listdir(meta_dir)])

for index in tqdm.tqdm(indices):
    class_path = os.path.join(root, 'img', 'class', '{}.png'.format(index))
    class_path_rgb = os.path.join(root, 'img', 'class_rgb', '{}.png'.format(index))

    class_img = np.array(mpimg.imread(class_path))

    class_img_rgb = colormap((class_img * 255.0).astype(int))

    mpimg.imsave(class_path_rgb, class_img_rgb)

