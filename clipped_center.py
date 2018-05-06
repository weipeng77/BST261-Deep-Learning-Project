#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:14:03 2018

@author: jingjingtang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:44:30 2018

@author: jingjingtang
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
import os
from skimage.io import imread, imread, imsave, imshow
import h5py
from scipy import misc
from seam_carver import intelligent_resize
import matplotlib.pyplot as plt
from resizeimage import resizeimage
from PIL import Image

base_h5 = os.path.join('mias-mammography', 'all_mias_scans.h5')
tif_dir = 'tiffs'
os.makedirs(tif_dir, exist_ok=True)
with h5py.File(base_h5, 'r') as f:
    mammo_df = pd.DataFrame(
        {k: v.value if len(v.shape)==1 else [sub_v for sub_v in v] 
         for k,v in f.items()}
    )
for k in mammo_df.columns:
    if isinstance(mammo_df[k].values[0], bytes):
        mammo_df[k] = mammo_df[k].map(lambda x: x.decode())


def to_path(c_row):
    out_path = os.path.join(tif_dir, '%s.tif' % c_row['REFNUM'])
    imsave(out_path, c_row['scan'])
    return out_path
mammo_df['scan'] = mammo_df.apply(to_path,1)
mammo_df['path'] = 'mias-mammography/all-mias/' + mammo_df['path'].astype(str)

# randomly assign X, Y
for i in mammo_df.index:
    if mammo_df['CLASS'][i] == 'NORM':
        mammo_df['X'][i], mammo_df['Y'][i] = 512, 512
        image_dir = os.path.join(mammo_df['path'][i])
        image = imread(image_dir)
        if sum(sum(image[512-150:512+150,512-150:512])) > sum(sum(image[512-150:512+150,512:512+150])):
            mammo_df['X'][i] = 512-75
        else:
            mammo_df['X'][i] = 512+75
        x = mammo_df['X'][i]
        if sum(sum(image[512-150:512,x-75:x+75])) > sum(sum(image[512:512+150,x-75:x+75])):
            mammo_df['Y'][i] = 512-75
        else:
            mammo_df['Y'][i] = 512+75
        mammo_df['RADIUS'][i] = 150
        mammo_df['SEVERITY'][i] = 0
#    else:                                  Then, 0 represents normal, 1 represents abnormal
#        mammo_df['SEVERITY'][i] = 1  
mammo_df = mammo_df.dropna(axis=0, how='any')
mammo_df['X'] = mammo_df['X'].astype(int)
mammo_df['Y'] = mammo_df['Y'].astype(int)
mammo_df['RADIUS'] = mammo_df['RADIUS'].astype(int)
#mammo_df = mammo_df.reset_index()



# clipped version of the original images with normal(random select center)
exist = {}
clipped_path = []
for i in mammo_df.index:
    image_dir = os.path.join(mammo_df['path'][i])
    image = Image.open(image_dir)
    image = image.crop(
            [mammo_df['X'][i]-mammo_df['RADIUS'][i], 
             1024-mammo_df['Y'][i]-mammo_df['RADIUS'][i],
             mammo_df['X'][i]+mammo_df['RADIUS'][i],
             1024-mammo_df['Y'][i]+mammo_df['RADIUS'][i]]
        )
    if mammo_df['REFNUM'][i] in exist.keys():
        exist[mammo_df['REFNUM'][i]] += 1
        write_dir = os.path.join('center_clipped', mammo_df['REFNUM'][i] + '_%d_clipped.tiff' % exist[mammo_df['REFNUM'][i]])
    else:
        exist[mammo_df['REFNUM'][i]] = 1
        write_dir = os.path.join('center_clipped', mammo_df['REFNUM'][i] + '_clipped.tiff')
    image = image.resize([48, 48])
    image.save(write_dir)
    clipped_path.append(write_dir)

mammo_df['clipped_path'] = clipped_path   # add path for newly clipped images