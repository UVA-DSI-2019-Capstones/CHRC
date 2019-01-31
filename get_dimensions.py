import csv
import os
import glob
import re
from pandas import DataFrame, Series
from openslide import open_slide
from PIL import Image
import timeit
import time
import math
import numpy as np



train_paths = ["/scratch/as3ek/CHRC_NEW_DATA/chrc_data/train/Celiac", "/scratch/as3ek/CHRC_NEW_DATA/chrc_data/valid/Celiac"]


images = {}
images_by_folder = {}
for train_path in train_paths:
    images_by_folder[str(train_path)] = []
    files = glob.glob(os.path.join(train_path, '*.svs'))
    for fl in files:
        flbase = os.path.basename(fl)
        flbase_noext = os.path.splitext(flbase)[0]
        images[flbase_noext]=fl
        images_by_folder[str(train_path)].append(flbase)
        
        
heights = {}
widths = {}        
        
for path in train_paths:
    
    for image in images_by_folder[path]:
        image = image.replace('.svs', '')
        heights[images[image]] = int(open_slide(images[image]).dimensions[1])
        widths[images[image]] = int(open_slide(images[image]).dimensions[0])
        
        
print(heights)
print(np.mean(list(heights.values())))
print(len(list(heights.values())))
print(widths)
print(np.mean(list(widths.values())))
print(len(list(widths.values())))