#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder 
    and dump them in a results folder """

import torch

from torchvision import datasets, transforms, utils
import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np
import argparse
import copy
import os
import cv2

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import *

from misc_functions import *
from classifier import train, parse_epoch
from roar_data_preparation import get_salience_based_adjusted_data
from utils import *

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

batch_size = 4
total_pixels = 244*244

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def experiment(criterion, optimizer, scheduler, cfg, percentages = [0.1, 0.3, 0.5, 0.7, 0.9]):
    
    # If adjusted data is not created, create it. 
    if not os.path.exists('dataset/cifar-10-adjusted'):
        print("Adjusted data is missing, so will now be created.")
        create_data(percentages, cfg)

    # Train model based on certrain adjusted data
    accuracy_list = perform_experiment(model, criterion, optimizer, scheduler, percentages, cfg)
    print(accuracy_list)
    # # Create plot
    # plt.plot(percentages, accuracy_list, marker = 'o')
    # plt.show()

def create_data(percentages, cfg):
    # Create train and test dataloader
    shuffle = True
    if config.num_classes == 10:
        transform = [CIFAR_10_TRANSFORM, CIFAR_10_TRANSFORM]
    else:
        transform = [CIFAR_100_TRANSFORM_TRAIN, CIFAR_100_TRANSFORM_TEST]

    train_loader = load_imageFolder_data(cfg.batch_size, transform[0], True, 2, cfg.data_dir, cfg.dataset_name, 
                                    train=True, name="cifar-10")
    test_loader = load_imageFolder_data(cfg.batch_size, transform[1], False, 2, cfg.data_dir, cfg.dataset_name, 
                                    train=False, name="cifar-10")

    # Get Ks
    Ks = [round((k * total_pixels)) for k in percentages]

    # Get adjusted data
    get_salience_based_adjusted_data(train_loader, ks, percentages, dataset = "train")
    get_salience_based_adjusted_data(test_loader, ks, percentages, dataset = "test")

def perform_experiment(model, criterion, optimizer, scheduler, percentages, cfg, num_classes = 10):
    accuracy_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    if num_classes == 10:
        transform = [CIFAR_10_TRANSFORM, CIFAR_10_TRANSFORM]
    else:
        transform = [CIFAR_100_TRANSFORM_TRAIN, CIFAR_100_TRANSFORM_TEST]

    for percentage in percentages:
        print(f"Training of model based on {percentage*100}% deletion of pixels.")
        copied_model = copy.deepcopy(model)

        data_dir = f"dataset/cifar-10-adjusted/cifar-{num_classes}-{percentage*100}%-removed/"
        adjusted_train_data = load_imageFolder_data(cfg.batch_size, transform[0], True, cfg.num_workers, data_dir + "train")
        adjusted_test_data = load_imageFolder_data(cfg.batch_size, transform[1], True, cfg.num_workers, data_dir + "test")
        
        train(copied_model, criterion, optimizer, scheduler, adjusted_train_data, adjusted_test_data, device,
        cfg.checkpoint_path, f"roar-{percentage*100}", cfg.epochs, cfg.save_epochs)

        eval_accuracy = parse_epoch(adjusted_test_data, copied_model, None, criterion, device, train=False)
        accuracy_list.append(eval_accuracy)
        print("Eval accur:", eval_accuracy)
        print("----------------------------------------------")

    return accuracy_list

if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_name', type=str, default="VGG-11", help="Name of the model when saved")
    parser.add_argument('--num_classes', type=int, default=100, help='Dimensionality of output sequence')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs until break')
    parser.add_argument('--load_model', type=str, default='', help='Give location of weights to load model')
    parser.add_argument('--save_epochs', type=int, default=1, help="save model after epochs")
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2, help='The amount of workers used to load data.')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda'")
    parser.add_argument('--save_model', type=bool, default=True, help="If set to false the model wont be saved.")
    parser.add_argument('--data_dir', type=str, default=PATH + 'dataset', help="data dir for dataloader")
    parser.add_argument('--dataset_name', type=str, default='/cifar-10-imageFolder', help= "Name of dataset contained in the data_dir")
    parser.add_argument('--checkpoint_path', type=str, default=PATH + 'saved-models/roar-models', help="model saving dir.")

    config = parser.parse_args()

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    config.checkpoint_path = os.path.join(config.checkpoint_path, '{model}-{epoch}-{type}.pth')


    device = torch.device(config.device)

    model = vgg11(pretrained=False, im_size = (3, 32, 32), num_classes=config.num_classes, class_size=512).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
    
    percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    experiment(criterion, optimizer, scheduler, config, percentages)

        




