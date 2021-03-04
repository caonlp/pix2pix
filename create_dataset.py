# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 01:35:37 2019

@author: Ayyoubzadeh
"""

base_path = './SOCOFing'
input_root_dir_x = base_path + '/Altered'
input_dir_y = base_path + '/Real'
data_file_name = base_path + '/data.npz'

import numpy as np
from numpy import array
import os
import cv2

image_shape = (100, 100, 1)


def get_images(images):
    images = array(images)
    return images


def normalize(input_data):
    return (input_data.astype(np.float32)) / 255.


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def load_files(ext):
    x_files = []
    y_files = []
    n = 0
    for d in os.listdir(input_root_dir_x):
        input_dir_x = os.path.join(input_root_dir_x, d)
        for f in os.listdir(input_dir_x):
            if f.upper().endswith(ext.upper()):
                img_path = os.path.join(input_dir_x, f)
                n += 1
                print(img_path)
                x_image = cv2.imread(img_path, 0)
                x_image = cv2.resize(x_image, (100, 100))
                if x_image.shape[0] == image_shape[0] and x_image.shape[1] == image_shape[1]:
                    x_image = np.reshape(x_image, image_shape)
                    y_file_path = os.path.join(input_dir_y,
                                               f.replace("_CR", "").replace("_Obl", "").replace("_Zcut", ""))
                    if os.path.exists(y_file_path):
                        x_files.append(x_image)
                        y_image = cv2.imread(y_file_path, 0)
                        y_image = cv2.resize(y_image, (100, 100))
                        y_image = np.reshape(y_image, image_shape)
                        y_files.append(y_image)
    return [x_files, y_files]


def save_data(ext, train_test_ratio=0.8):
    [x_files, y_files] = load_files(ext)
    x = get_images(x_files)
    x = normalize(x)
    y = get_images(y_files)
    y = normalize(y)
    np.savez(data_file_name, x=x, y=y)


save_data('.bmp')
