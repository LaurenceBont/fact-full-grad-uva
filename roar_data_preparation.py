import torch

from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import cv2

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import *
from models.resnet import *
from misc_functions import *
from utils import *

def compute_saliency_and_save_image(sample_loader, ks, dataset = "train"):
    image_counter = 0

    for data, target in sample_loader:
        data, target = data.to(device).requires_grad_(), target.to(device)

        # Compute saliency maps for the input data.
        cam, _ = fullgrad.saliency(data)

        # Find most important pixels and replace.
        for k in ks: 

            # Get unnormalized image and heat map.
            sal_map = cam[i,:,:,:].squeeze()
            image = unnormalize(data[i,:,:,:])

            # Get k indices and replace within image
            indices = return_k_index_argsort(sal_map.detach().numpy(), k, "roar")
            new_image = replace_pixels(image, indices, 'zero')

            # Save adjusted images, if needed.
            utils.save_image(new_image, f'pixels_removed/{method}/removal{k*100}%/img_id={image_counter}removal={k*100}%.jpeg')

            image_counter += 1
    image_counter = 0

def train_model():

    retu


if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_name', type=str, default="VGG-11", help="Name of the model when saved")
    parser.add_argument('--num_classes', type=int, default=100, help='Dimensionality of output sequence')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs until break')
    parser.add_argument('--load_model', type=str, default='', help='Give location of weights to load model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda'")
    parser.add_argument('--save_model', type=bool, default=True, help="If set to false the model wont be saved.")
    parser.add_argument('--data_dir', type=str, default=PATH + 'dataset', help="data dir for dataloader")
    parser.add_argument('--dataset_name', type=str, default='/cifar-100-imageFolder', help= "Name of dataset contained in the data_dir")
    parser.add_argument('--checkpoint_path', type=str, default=PATH + 'saved-models/', help="model saving dir.")

    config = parser.parse_args()

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    config.checkpoint_path = os.path.join(config.checkpoint_path, '{model}-{epoch}-{type}.pth')

    # Create dataloaders
    train_loader = load_cifar_data(1, CIFAR_100_TRANSFORM_TRAIN,
                                True, 2, config.data_dir, config.dataset_name, train=True)
    test_loader = load_cifar_data(1, CIFAR_100_TRANSFORM_TEST,
                                False, 2, config.data_dir, config.dataset_name, train=False)

    
    # Get sample and total pixels 
    sample_img = next(iter(train_loader))[0]
    total_pixels = sample_img.shape[2] * sample_img.shape[3]

    # Calculate k's based on percentages and total pixels
    percentages = [0.001, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1]
    Ks = [round((k * total_pixels)) for k in percentages]

    # uncomment to use VGG
    model = vgg16_bn(pretrained=True, img_size = sample_img.shape).to(device)
    # model = resnet18(pretrained=True).to(device)

    # Initialize FullGrad objects
    fullgrad = FullGrad(model, im_size = sample_img.squeeze().shape)

    # Create ROAR images
    compute_saliency_and_save_image(train_loader, Ks, "train")
    compute_saliency_and_save_image(test_loader, Ks, "test")
