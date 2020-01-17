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
from classifier import train

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])

def create_data_dirs(percentages):
    """
        Creates directories to save adjusted images.

        percentages : the percentages of pixels which are adjusted, used to create
                        convenient directory names.
    """
    for percentage in percentages:
        directory = f'dataset/cifar-100-adjusted/cifar-100-{percentage*100}%-removed' 
        create_cifar_dir(directory)

def create_adjusted_image_and_save(ks, percentages):
    """
        Creates adjusted images based on different K's, and saves them.

        ks          : amount of pixels which are adjusted within the image.
        percentages : the percentages of pixels which are adjusted, used to save images in correct
                      directories.
    """
    for k, percentage in zip(ks, percentages): 

            # Get unnormalized image and heat map.
            sal_map = cam.squeeze()
            image = unnormalize(data.squeeze())

            # Get k indices and replace within image
            indices = return_k_index_argsort(sal_map.detach().numpy(), k, method)
            new_image = replace_pixels(image, indices, approach = approach)

            # Save adjusted images
            data_dir = f'dataset/cifar-100-adjusted/cifar-100-{percentage*100}%-removed'
            save_cifar_image(data_dir, target, new_image, "cifar-100", dataset)


def get_salience_based_adjusted_data(sample_loader, ks, percentages, dataset = "train"):
    """
        Creates adjusted images based on different K's, and saves them.
        
        sample_loader: created dataloader, used to sample the images.
        ks           : amount of pixels which are adjusted within the image.
        percentages  : the percentages of pixels which are adjusted, used to save images in correct
                       directories.
        dataset      : Used to define which set is used.  
    """
    image_counter = 0
    method = "roar"
    approach = "zero"
    create_data_dirs(percentages)

    for data, target in sample_loader:
        data, target = data.to(device).requires_grad_(), target.to(device)

        # Compute saliency maps for the input data.
        cam, _ = fullgrad.saliency(data)

        # Find most important pixels, replace and save adjusted image.
        create_adjusted_image_and_save(ks, percentages)


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

    # Load or train model
    if os.path.exists('saved-models/vgg11-60-best.pth'):
        print("The model will now be loaded.")
        model.load_state_dict(torch.load('saved-models/vgg11-60-best.pth'), True if device == 'cuda' else False)
    else:
        model = vgg11(pretrained=False, im_size = sample_img.shape, num_classes=config.num_classes, class_size=512).to(device)

        # Train model on cifar-100
        print("The model will now be trained.")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

        train(model, criterion, optimizer, scheduler, train_loader, test_loader, device,
            config.checkpoint_path, config.model_name, config.epochs, config.save_epochs)

    # Initialize FullGrad objects
    fullgrad = FullGrad(model, im_size = sample_img.squeeze().shape)

    # Create ROAR images
    print("The adjusted data will now be created.")
    get_salience_based_adjusted_data(train_loader, Ks, percentages, "train")
    get_salience_based_adjusted_data(test_loader, Ks, percentages, "test")
