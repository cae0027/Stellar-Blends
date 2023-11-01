import  numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from functools import partial


cutout_files = pd.read_csv('./stars_and_blends.csv')
cutout_files = cutout_files.sample(frac=1).reset_index()
cutout_files.head()

# Initialize DataFrame to append each cutout data to
star_blend = pd.DataFrame({})

# Define lists that each minimum and maximum pixel value will be appended to
min_pixel = []
max_pixel = []

# Create a DataFrame to append plotting data to
plot_data = pd.DataFrame({})

for idx, row in tqdm(cutout_files.iterrows(),total=cutout_files.shape[0]): 
    obj_id = row['class']
         
    data_flattened = row.iloc[3:]
    
    min_pixel.append(np.min(data_flattened))
    max_pixel.append(np.max(data_flattened))
    
    star_blend = star_blend.append([np.append(obj_id, data_flattened)], ignore_index=True)

plot_data = pd.concat([plot_data, pd.DataFrame({'raw': data_flattened.to_numpy()})], axis=1)
star_blend.to_csv('raw_image_data.csv', header=False, index=False)


min_pixel_all = np.min(np.abs(min_pixel))
max_pixel_all = np.max(np.abs(max_pixel))

def norm_1(data):
    """
    For each realization:
    Take the log of each pixel value
    Find the minimum pixel value accross the image
    Subtract that minimum value from each pixel
    """
    data_log = np.log10(np.abs(data))
    min_log_data = np.amin(data_log)
    data_norm = data_log - min_log_data
    return data_norm

def norm_2(data):
    """
    For each cutout:
    Calculate minimum pixel value across image
    Calculate maximum pixel value accross image
    Scale all data between (0, 1) with:
    𝑛𝑜𝑟𝑚 = (𝑑𝑎𝑡𝑎−𝑚𝑖𝑛) / (𝑚𝑎𝑥−𝑚𝑖𝑛)
    """
    min_data = np.min(data)
    max_data = np.max(data)
    data_norm = (data - min_data) / (max_data - min_data)
    return data_norm
 
def norm_21(data):
    """
    Composed norm_1 with norm_1
    """
    min_data = np.min(norm_1(data))
    max_data = np.max(norm_1(data))
    data_norm = (norm_1(data) - min_data) / (max_data - min_data)
    return data_norm

def norm_3(data):
    """
    The same as technique 2, but now min and max are the minimum and maximum pixel value over ALL images
    For each cutout:
    Scale all data between (0, 1) with:
           𝑛𝑜𝑟𝑚 = (𝑑𝑎𝑡𝑎-𝑚𝑖𝑛𝑎𝑙𝑙)/ (𝑚𝑎𝑥𝑎𝑙𝑙-𝑚𝑖𝑛𝑎𝑙𝑙)
    """
    data_norm = (data - min_pixel_all) / (max_pixel_all - min_pixel_all)
    return data_norm

def norm_31(data):
    data_norm = (norm_1(data) - min_pixel_all) / (max_pixel_all - min_pixel_all)
    return data_norm

def norm_4(data):
    """
    For each cutout:
    Find the minimum pixel value in the image
    Subtract that value off of each pixel
    Divide each pixel in the image by the maximum value over ALL images
    """
    min_data = np.amin(data)
    data_min_subtracted = data - min_data
    data_norm = data_min_subtracted/max_pixel_all
    return data_norm

# Compose norm_4 and norm_1
def norm_41(data):
    data_norm = norm_4(norm_1(data))
    return data_norm

# exponentiate the result from norm_41
def norm_51(data):
    data_norm = np.exp(norm_4(data))
    return np.exp(data_norm)

def norm_5(data):
    """
    The same as technique 4, but now we take the log of each value first
    For each cutout:
    Log each pixel (and the maximum pixel value)
    Find the minimum pixel value in the image
    Subtract that value off of each pixel
    Divide each pixel in the image by the maximum value over ALL images

    """
    log_data = np.log10(np.abs(data))
    log_max_pixel = np.log10(np.abs(max_pixel_all))
    min_data = np.amin(log_data)
    data_min_subtracted = log_data - min_data
    data_norm = data_min_subtracted/log_max_pixel
    return data_norm

def nthroot(data, r=0.2):
    """
    For each realization:
    take the rth root of each pixel
    if the rth root is odd, it takes care of -ve pixel values
    normalize by applying norm_2
    """
    nthrooth = (data)**(r)
    data_norm = nthrooth / np.amax(nthrooth)
    return data_norm

# log values then nth root
def nthroot_log(data, r=0.2):
    # data_norm = (np.abs(data))**(r)
    data_norm = norm_2((norm_1(abs(data)))) 
    return data_norm**r


techniques = {
              'norm_1':norm_1, 'norm_2':norm_2, 'norm_3':norm_3, 'norm_4':norm_4, 
              'norm_5':norm_5,'norm_21':norm_21, 'norm_31':norm_31, 'norm_41':norm_41,
              'norm_51':norm_51,  
              #'nthroot':nthroot,
              'nthroot_log':nthroot_log,
             }
# get more rth roots normalization
roots = np.linspace(0,1,20)
rts = {f"nthroot_{rt}": partial(nthroot,  r=rt) for rt in roots}
# merge the two dictionaries
techniques = {**rts, **techniques}

# store names of the normalized csv files in
norm_data_names = []
for num, technique in enumerate(techniques):
    star_blend_norm = pd.DataFrame({})
    for idx, row in tqdm(star_blend.iterrows(), total=star_blend.shape[0], desc='Technique '+ str(num+1), leave=True):
        # Separate type and data
        raw_data = row[1:].values
        obj_id = row[:1].values 

        # Run raw data through each normalization technique
        norm_data = techniques[technique](raw_data)

        # Append values for current cutout to dataframe
        star_blend_norm = star_blend_norm.append([np.append(obj_id, norm_data)], ignore_index=True)

    # Append plot data for first 5 images
    plot_data = pd.concat([plot_data, pd.DataFrame({technique: star_blend_norm[:5].to_numpy().flatten()})], axis=1)

    # Save to .csv
    star_blend_norm.to_csv('./data-norm/' + technique + '_data.csv', header=False, index=False)
    norm_data_names.append('./data-norm/' + technique + '_data.csv')
        
