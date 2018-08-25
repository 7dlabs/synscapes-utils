#! /usr/bin/env python3

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from helpers import *

# Arguments ---

parser = argparse.ArgumentParser(
    description='Displays bounding boxes over SynScapes images.')

parser.add_argument('path',
                    help='Path to SynScapes root directory.')
parser.add_argument('--index', '-i', type=int, required=True,
                    help='Index of image to visualize')
parser.add_argument('--type', choices=['2d', '3d'], default='2d',
                    help='Whether to visualize 2d or 3d bounding boxes')
parser.add_argument('--threshold', type=float, default=90,
                    help='Percentage of occlusion and truncation at which to cull bounding boxes.')
parser.add_argument('--xkcd', action='store_true',
                    help='Yes.')

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

# Functions ---


def draw_2d():
    for inst, box in bbox_data.items():
        # Info
        class_idx = metadata['instance']['class'][inst]
        class_name = class_id_to_str[class_idx]
        occluded = metadata['instance']['occluded'][inst]
        if occluded > args.threshold / 100.0:
            continue
        # Box
        xmin = box['xmin']
        xmax = box['xmax']
        ymin = box['ymin']
        ymax = box['ymax']
        c = (xmin * w, ymin * h)
        width = (xmax - xmin) * w
        height = (ymax - ymin) * h
        # Color
        color = tableau20[class_idx % len(tableau20)]
        # Draw
        patch = plt.Rectangle(c, width, height, linewidth=3, edgecolor=color, fill=False)
        ax.add_patch(patch)
        # Text
        plt.text(c[0], c[1] - 2, '{}'.format(class_name),
                 color=text_color, bbox=dict(facecolor=color, alpha=0.5))


def draw_3d():
    vehicle_to_sensor_xform = vehicle_to_sensor_transform(metadata)
    # Projection
    fx = metadata['camera']['intrinsic']['fx']
    fy = metadata['camera']['intrinsic']['fy']
    u0 = metadata['camera']['intrinsic']['u0']
    v0 = metadata['camera']['intrinsic']['v0']
    # Transform corner points to image space
    is_pts = {}
    for inst, box in bbox_data.items():
        # Cull
        occluded = metadata['instance']['occluded'][inst]
        if occluded > args.threshold / 100.0:
            continue
        # Vehicle coordinates
        vc = box['origin']
        # Add zero vector for origin offset
        box['o'] = (0, 0, 0)
        # Image coordinates
        is_pts[inst] = {
            name: vehicle_space_to_image_space(vec_sum(box['origin'], box[name]),
                                               vehicle_to_sensor_xform,
                                               fx, fy, u0, v0) \
            for name in ['o', 'x', 'y', 'z']
        }
    # Create lines
    lines = []
    for inst, box in is_pts.items():
        # Color
        class_idx = metadata['instance']['class'][inst]
        class_name = class_id_to_str[class_idx]
        color = tableau20[class_idx % len(tableau20)]
        # Vertices
        o = box['o']
        x = box['x']
        y = box['y']
        z = box['z']
        # Lines
        for tgt in [x, y, z]:
            verts_x = [o[0], tgt[0]]
            verts_y = [o[1], tgt[1]]
            line = plt.Line2D(verts_x, verts_y, c=color, lw=2)
            ax.add_line(line)
            # plt.scatter(verts_x, verts_y, c=(1, 1, 1))
        # Text
        plt.text(o[0] + 2, o[1] - 2, '{}'.format(class_name),
                 color=text_color, bbox=dict(facecolor=color, alpha=0.5))
        
    
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

img_path = os.path.join(root, 'img', 'rgb', '{}.png'.format(args.index))
meta_path = os.path.join(root, 'meta')

metadata = read_metadata(meta_path, args.index)[args.index]
bbox_data = metadata['instance']['bbox{}'.format(args.type)]

img = mpimg.imread(img_path)
w = float(img.shape[1])
h = float(img.shape[0])

text_color = (0.9, 0.9, 0.9)
if args.xkcd:
    text_color = (0, 0, 0)
    plt.xkcd()

fig, ax = plt.subplots(1)
dpi = fig.get_dpi()

# Add image background
ax.imshow(img)

# Add boxes
if args.type == '2d':
    draw_2d()
elif args.type == '3d':
    draw_3d()

# Display
fig.set_size_inches(w / dpi, h / dpi, forward=True)
plt.axis('off')
plt.tight_layout()
plt.show()
