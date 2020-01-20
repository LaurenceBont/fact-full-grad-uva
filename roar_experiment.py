#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder 
    and dump them in a results folder """

import torch

from torchvision import datasets, transforms, utils
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import copy
import os
import cv2

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import *
from models.resnet import *
from misc_functions import *
from roar_data_preparation import get_salience_based_adjusted_data
from utils import load_imageFolder_data, CIFAR_100_TRANSFORM_TRAIN, CIFAR_100_TRANSFORM_TEST

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

batch_size = 4
total_pixels = 244*244

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def experiment(criterion, optimizer, scheduler, cfg, percentages = [0.10, 0.33, 0.66, 0.99]):
    
    # If adjusted data is not created, create it. 
    if not os.path.exists('dataset/cifar-100-adjusted'):
        get_salience_based_adjusted_data(sample_loader, ks, percentages, dataset = "train")
        get_salience_based_adjusted_data(sample_loader, ks, percentages, dataset = "test")

    # Train model based on certrain adjusted data
    accuracy_list = do_experiment(model, criterion, optimizer, scheduler, percentages, cfg)

    # Create plot
    plt.plot(percentages, accuracy_list, marker = 'o')
    plt.show()

def do_experiment(model, criterion, optimizer, scheduler, percentages, cfg):
    accuracy_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for percentage in percentages:
        copied_model = copy.deepcopy(model)

        data_dir = f"dataset/cifar-100-adjusted/cifar-100-{percentage}%-removed/"
        adjusted_train_data = load_imageFolder_data(cfg.batch_size, CIFAR_100_TRANSFORM_TRAIN, cfg.shuffle, cfg.num_workers, data_dir + "train")
        adjusted_test_data = load_imageFolder_data(cfg.batch_size, CIFAR_100_TRANSFORM_TEST, cfg.shuffle, cfg.num_workers, data_dir + "test")
        
        train(copied_model, criterion, optimizer, scheduler, adjusted_train_data, adjusted_test_data, device,
        checkpoint_path, model_name, epochs, save_epochs)

        accuracy_list.append(parse_epoch(adjusted_test_data, model_k_data, None, criterion, device, train=False))

if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_name', type=str, default="VGG-11", help="Name of the model when saved")
    parser.add_argument('--num_classes', type=int, default=100, help='Dimensionality of output sequence')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs until break')
    parser.add_argument('--load_model', type=str, default='', help='Give location of weights to load model')
    parser.add_argument('--save_epochs', type=int, default=1, help="save model after epochs")
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda'")
    parser.add_argument('--save_model', type=bool, default=True, help="If set to false the model wont be saved.")
    parser.add_argument('--data_dir', type=str, default=PATH + 'dataset', help="data dir for dataloader")
    parser.add_argument('--dataset_name', type=str, default='/cifar-100-imageFolder', help= "Name of dataset contained in the data_dir")
    parser.add_argument('--checkpoint_path', type=str, default=PATH + 'saved-models/', help="model saving dir.")

    config = parser.parse_args()

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    config.checkpoint_path = os.path.join(config.checkpoint_path, '{model}-{epoch}-{type}.pth')


    device = torch.device(config.device)

    model = vgg11(pretrained=False, im_size = (3, 32, 32), num_classes=config.num_classes, class_size=512).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
    
    percentages = [0.10, 0.33, 0.66, 0.99]
    experiment(criterion, optimizer, scheduler, config, percentages)
        




