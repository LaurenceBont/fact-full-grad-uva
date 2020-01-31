"""
    This file creates adjusted datasets using the saliency map
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils

from classifier import train
# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from utils import (UNNORMALIZE, create_imagefolder_dir, load_data,
                   replace_pixels, return_k_index_argsort,
                   save_imagefolder_image)


def create_data(percentages, model_cfg, loader_cfg, salience_method="full_grad"):
    # Create train and test dataloader
    train_loader = load_data(1, loader_cfg.transform, False, 1, loader_cfg.data_dir, loader_cfg.dataset, train=True, name=loader_cfg.dataset)
    test_loader = load_data(1, loader_cfg.transform, False, 1, loader_cfg.data_dir, loader_cfg.dataset, train=False, name=loader_cfg.dataset)
    
    # number of pixels in k percent of the image
    num_pixel_list = [round((percentage * loader_cfg.image_size)) for percentage in percentages]


    # Get adjusted data
    create_salience_based_adjusted_data(train_loader, num_pixel_list, percentages, model_cfg.device, salience_method, dataset="train")
    create_salience_based_adjusted_data(test_loader, num_pixel_list, percentages, model_cfg.device, salience_method, dataset="test")

def create_data_dirs(percentages, num_classes, salience_method):
    """
        Creates directories to save adjusted images.

        percentages : the percentages of pixels which are adjusted, used to create
                        convenient directory names.
    """
    
    os.mkdir(f'dataset/roar_{salience_method}')

    for percentage in percentages:
        directory = f'dataset/roar_{salience_method}/cifar-{num_classes}-{percentage*100}%-removed' 
        create_imagefolder_dir(directory, num_classes)

def create_adjusted_images_and_save(idx, data, cam, target, ks, percentages, num_classes, dataset, method, approach = "zero"):
    """
        Creates adjusted images based on different K's, and saves them.
        
        cam         : Salience map of most important pixels.
        ks          : amount of pixels which are adjusted within the image.
        percentages : the percentages of pixels which are adjusted, used to save images in correct
                      directories.
    """
    sal_map = cam.squeeze()

    for k, percentage in zip(ks, percentages): 

        # Get unnormalized image
        image = UNNORMALIZE(data.squeeze())

        # Get k indices and replace within image
        indices = return_k_index_argsort(sal_map.cpu().detach().numpy(), k, method)
        new_image = replace_pixels(image, indices, approach = approach)

        # Save adjusted images
        data_dir = f'dataset/roar_{method}/cifar-{num_classes}-{percentage*100}%-removed'
        save_imagefolder_image(data_dir, target, new_image, idx, dataset)


def create_salience_based_adjusted_data(sample_loader, ks, percentages, device, salience_method="full_grad", num_classes=10, dataset="train", method="roar"):
    """
        Creates adjusted images based on different K's, and saves them.
        
        sample_loader: created dataloader, used to sample the images.
        ks           : amount of pixels which are adjusted within the image.
        percentages  : the percentages of pixels which are adjusted, used to save images in correct
                       directories.
        dataset      : Used to define which set is used.  
    """

    # Creates data directories if needed.
    if not os.path.exists(f'dataset/roar_{salience_method}'):
        create_data_dirs(percentages, num_classes, salience_method)
    else:
        print(f"{dataset}set already created!")

    # Loops over sample loader to creates per sample every adjusted image, and saves them.
    for idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        if salience_method == "full_grad":
            _, salience_map, _ = fullgrad.saliency(data)

        elif salience_method == "input_grad":
            salience_map, _, _ = fullgrad.saliency(data)

        elif salience_method == "random":
            salience_map = None
            method = "random"
            
        # Find most important pixels, replace and save adjusted image.
        create_adjusted_images_and_save(idx, data, salience_map, target, ks, percentages, num_classes, dataset, method)