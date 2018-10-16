import staintools
import csv
import os
import glob
import re
from pandas import DataFrame, Series
from PIL import Image
import timeit
import time
import cv2
from matplotlib import pyplot as plt


train_paths = ["../data_segments/data_balanced_duplicate_sample/train/EE/"]

# get images
images = {}
images_by_folder = {}
for train_path in train_paths:
    images_by_folder[str(train_path)] = []
    files = glob.glob(os.path.join(train_path, '*.jpg'))
    for fl in files:
        flbase = os.path.basename(fl)
        flbase_noext = os.path.splitext(flbase)[0]
        images[flbase_noext]=fl
        images_by_folder[str(train_path)].append(flbase_noext)
        
# initialize stain and brightness normalizer
stain_normalizer = staintools.StainNormalizer(method='vahadane')
standardizer = staintools.BrightnessStandardizer()

# choose target image
target_image = staintools.read_image("../data_segments/data_balanced_duplicate_sample/valid/Celiac/C06-27_01_1000_2000_0.jpg")
standard_target_image = standardizer.transform(target_image)
stain_normalizer.fit(standard_target_image)

# get destination path
path_change_map = {}

for key in list(images_by_folder.keys()):
    temp = key.replace('..', '/scratch/as3ek')
    path_change_map[key] = temp
    
    
count = 0
for key in images_by_folder.keys():
    for value in images_by_folder[key]:
        count += 1
        if count < 3630:
            continue
        try:
            print (str(count) + ' ' + str(value))
            source_img_path = str(key) + str(value) + '.jpg'
            dest_img_path = str(path_change_map[key]) + str(value) + '.jpg'
            img = staintools.read_image(source_img_path)
            # standardize brightness
            img_standard = standardizer.transform(img)
            # transform the images
            img_normalized = stain_normalizer.transform(img_standard)
            # write image to path

    #             plt.imshow(img)
    #             plt.title('my picture')
    #             plt.show()
    #             plt.imshow(img_normalized)
    #             plt.title('my picture')
    #             plt.show()

            if (count % 10000 == 0):
                plt.imshow(img)
                plt.title('my picture')
                plt.show()
                plt.imshow(img_normalized)
                plt.title('my picture')
                plt.show()
                print(value + ' Saved! Count: ' + str(count) + '  \n ' + str(dest_img_path))
            cv2.imwrite(os.path.normpath(dest_img_path), img_normalized)
        except:
            continue
