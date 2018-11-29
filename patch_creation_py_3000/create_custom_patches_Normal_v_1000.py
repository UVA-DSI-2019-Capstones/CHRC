
# coding: utf-8

# In[1]:

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
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk


# In[2]:

train_paths = ["/scratch/ss4yd/chrc_data/valid/Normal/"]

images = {}
images_by_folder = {}
for train_path in train_paths:
    images_by_folder[str(train_path)] = []
    files = glob.glob(os.path.join(train_path, '*.svs'))
    for fl in files:
        flbase = os.path.basename(fl)
        flbase_noext = os.path.splitext(flbase)[0]
        images[flbase_noext] = fl
        images_by_folder[str(train_path)].append(flbase)


# In[3]:

path_change_map = {}

for key in list(images_by_folder.keys()):
    temp = key.replace('chrc_data', 'chrc_data_patches_3000')
    path_change_map[key] = temp


# In[5]:

# images_by_folder


# In[13]:

def convert_to_3d_array(patch):
    rgb = patch.convert('RGB')
    return np.array(rgb)


def optical_density(tile):
    tile = tile.astype(np.float64)
    od = -np.log((tile+1)/240)
    return od


def keep_tile(tile, tile_size,  tissue_threshold):
    if tile.shape[0:2] == (tile_size, tile_size):
        # print("inside if")
        tile_orig = tile
        tile = rgb2gray(tile)
        tile = 1 - tile

        tile = canny(tile)

        tile = binary_closing(tile, disk(10))
        tile = binary_dilation(tile, disk(10))
        tile = binary_fill_holes(tile)
        percentage1 = tile.mean()

        check1 = percentage1 >= tissue_threshold

        # Check 2
        # Convert to optical density values
        tile = optical_density(tile_orig)
        # Threshold at beta
        beta = 0.15
        tile = np.min(tile, axis=2) >= beta
        # Apply morphology for same reasons as above.
        tile = binary_closing(tile, disk(2))
        tile = binary_dilation(tile, disk(2))
        tile = binary_fill_holes(tile)
        percentage2 = tile.mean()
        check2 = percentage2 >= tissue_threshold
        # print(percentage1, percentage2)
        result = check1 and check2
        return result, percentage1, percentage2
    else:
        return False, -1, -1


def process_tile(tile, sample_size, grayscale, slide_num):
    """
    Process a tile into a group of smaller samples.
    Cut up a tile into smaller blocks of sample_size x sample_size pixels,
    change the shape of each sample from (H, W, channels) to
    (channels, H, W), then flatten each into a vector of length
    channels*H*W.
    Args:
      tile_tuple: A (slide_num, tile) tuple, where slide_num is an
        integer, and tile is a 3D NumPy array of shape
        (tile_size, tile_size, channels).
      sample_size: The new width and height of the square samples to be
        generated.
      grayscale: Whether or not to generate grayscale samples, rather
        than RGB.
    Returns:
      A list of (slide_num, sample) tuples representing cut up tiles,
      where each sample is a 3D NumPy array of shape
      (sample_size_x, sample_size_y, channels).
    """
    if grayscale:
        tile = rgb2gray(tile)[:, :, np.newaxis]  # Grayscale
        # Save disk space and future IO time by converting from [0,1] to [0,255],
        # at the expense of some minor loss of information.
        tile = np.round(tile * 255).astype("uint8")
    x, y, ch = tile.shape
    # 1. Reshape into a 5D array of (num_x, sample_size_x, num_y, sample_size_y, ch), where
    # num_x and num_y are the number of chopped tiles on the x and y axes, respectively.
    # 2. Swap sample_size_x and num_y axes to create
    # (num_x, num_y, sample_size_x, sample_size_y, ch).
    # 3. Combine num_x and num_y into single axis, returning
    # (num_samples, sample_size_x, sample_size_y, ch).
    samples = (tile.reshape((x // sample_size, sample_size, y // sample_size, sample_size, ch))
                   .swapaxes(1, 2)
                   .reshape((-1, sample_size, sample_size, ch)))
    samples = [(slide_num, sample) for sample in list(samples)]
    return samples


def create_patches(img_path, patch_size, overlap):

    img_name = img_path.split('/')[-1].split('.')[0]
    print(img_name)
    # img_path = image.values()

    slide = open_slide(str(img_path))

    # steps to advance per axis with overlap
    step_size = patch_size - overlap

    # get dimensions of the image
    xlim = slide.level_dimensions[0][0]
    ylim = slide.level_dimensions[0][1]
    print("Dimensions x: " + str(xlim) + " y: " + str(ylim))

    # get the number of times to traverse each axis
    x_ind = math.ceil(xlim/(step_size))
    y_ind = math.ceil(ylim/(step_size))

    # initialize list to store patches
    patches = {}
    patches_dict = {}
    patches_vals = []

    # pixels left to traverse in the y-axis at the beginning of the traversal
    img_y_left = ylim
    # initialize the starting y corner
    y = 0 - step_size
    for y_ in range(y_ind):
        # patches_dict = {}
        # patches_vals = []
        # initialize the starting x corner
        x = 0-step_size

        # advance the y axis (note: it starts with 0)
        y = y + step_size

        # pixels left to traverse in the x-axis at the beginning of the traversal
        img_x_left = xlim

        # update the number of pixels left to traverse
        img_y_left = img_y_left - step_size*(bool(y_)) - patch_size*(not bool(y_))
        # print('y - left: ' + str(img_y_left))

        # if more than patch size left, get the (patch_size x patch_size) image
        if (img_y_left > 0 and img_y_left > step_size):
            for x_ in range(x_ind):
                x = x + (step_size)
                img_x_left = img_x_left - step_size*(bool(x_)) - patch_size*(not bool(x_))
                # print(img_x_left)
                if (img_x_left > 0 and img_x_left > step_size):
                    img_name_key = img_name + "___"+str(x)+"_"+str(y)
                    patches_dict[img_name_key] = slide.read_region(
                        (x, y), 0, (patch_size, patch_size))
                    # patches_vals.append(slide.read_region((x, y), 0, (patch_size, patch_size)))
                    # patches[img_name]=patches_vals
                elif (img_x_left < step_size and img_x_left > 0):
                    x = xlim - patch_size
                    img_name_key = img_name + "___"+str(x)+"_"+str(y)
                    patches_dict[img_name_key] = slide.read_region(
                        (x, y), 0, (patch_size, patch_size))
                    # patches_vals.append(slide.read_region((x, y), 0, (patch_size, patch_size)))
                    # patches[img_name]=patches_vals
                    break
                # patches[img_name]=patches_vals
        # if less than patch size left, get the rest of the image, regardless of the overlap
        elif (img_y_left > 0 and img_y_left < step_size):
            y = ylim - patch_size
            for x_ in range(x_ind):
                x = x + (step_size)
                img_x_left = img_x_left - step_size*(bool(x_)) - patch_size*(not bool(x_))
                if (img_x_left > 0 and img_x_left > step_size):
                    img_name_key = img_name + "___"+str(x)+"_"+str(y)
                    patches_dict[img_name_key] = slide.read_region(
                        (x, y), 0, (patch_size, patch_size))
                    # patches_vals.append(slide.read_region((x, y), 0, (patch_size, patch_size)))
                    # patches[img_name]=patches_vals
                elif (img_x_left < step_size and img_x_left > 0):
                    x = xlim - patch_size
                    img_name_key = img_name + "___"+str(x)+"_"+str(y)
                    patches_dict[img_name_key] = slide.read_region(
                        (x, y), 0, (patch_size, patch_size))
                    # patches_vals.append(slide.read_region((x, y), 0, (patch_size, patch_size)))
                    # patches[img_name]=patches_vals
                    break
            break

    return patches_dict


# In[5]:

def create_filtered_patches(slide, patch_size, overlap, slide_num):

    patches = create_patches(slide, patch_size, overlap)
    patchl = []
    # convert to RGB from RGBA
    for key in patches.keys():
        patchl.append(patches[key])
    tiles = [convert_to_3d_array(patch) for patch in patchl]

    filtered_tiles = [
        tile for tile in tiles if keep_tile(tile, patch_size, 0.20)]

    samples = [process_tile(tile, patch_size, False, slide_num)
               for tile in filtered_tiles]

    return samples


def save_nonlabelled_sample_2_jpeg(sample, save_dir):
    """
    Save the sample without labels into JPEG
    Args:
      sample_element: a sample tuple without labels, e.g. (slide_num, sample)
      save_dir: the file directory at which to save JPEGs
    """
    slide_num, img_value = sample
    filename = '{slide_num}_{hash}.jpeg'.format(
        slide_num=slide_num, hash=np.random.randint(1e4))
    filepath = os.path.join(save_dir, filename)
    save_jpeg_help(img_value, filepath)


def save_jpeg_help(img_value, filepath):
    """
     Save data into JPEG
     Args:
       img_value: the image value with the size (img_size_x, img_size_y, channels)
       file path: the file path at which to save JPEGs
     """
    dir = os.path.dirname(filepath)
    os.makedirs(dir, exist_ok=True)
    img = Image.fromarray(img_value.astype(np.uint8), 'RGB')
    img.save(filepath)


# In[16]:

patch_size = 1000
overlap = 500
threshold = 0.3

score_df = DataFrame()
imgName = []
score1L = []
score2L = []

for key in images.keys():
    img_path = images[key]
    img_save_path = img_path.replace(
        'chrc_data_patches_norm', 'chrc_data_patches_norm_1000')
    img_save_path2 = img_save_path.replace(img_save_path.split('/')[-1], '')
    # print(img_save_path)
    patches = create_patches(
        img_path=img_path, patch_size=patch_size, overlap=overlap)

    for key2 in patches.keys():

        img_name = key2
        img_save_path_n = img_save_path2+str(img_name)
        # print(img_save_path)
        patch = patches[key2]

        tile = convert_to_3d_array(patch)

        ifKeep, score1, score2 = keep_tile(tile, patch_size, threshold)
        print((ifKeep, score1, score2))
        imgName.append(img_name)
        score1L.append(score1)
        score2L.append(score2)

        if(ifKeep):
            patch.save(img_save_path_n + '.jpg')


imgNameS = Series(imgName)
score1S = Series(score1L)
score2S = Series(score2L)

score_df['img_name'] = imgNameS
score_df['score1'] = score1S
score_df['score2'] = score2S

score_df.to_csv('/scratch/ss4yd/Normal_v_1000.csv')
