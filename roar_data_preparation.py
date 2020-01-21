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

def create_data_dirs(percentages, num_classes):
    """
        Creates directories to save adjusted images.

        percentages : the percentages of pixels which are adjusted, used to create
                        convenient directory names.
    """
    
    os.mkdir(f'dataset/cifar-{num_classes}-adjusted')

    for percentage in percentages:
        directory = f'dataset/cifar-{num_classes}-adjusted/cifar-{num_classes}-{percentage*100}%-removed' 
        create_imagefolder_dir(directory, num_classes)

def create_adjusted_images_and_save(idx, data, cam, target, ks, percentages, num_classes, dataset, method = "roar", approach = "zero"):
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
            image = unnormalize(data.squeeze())

            # Get k indices and replace within image
            indices = return_k_index_argsort(sal_map.detach().numpy(), k, method)
            new_image = replace_pixels(image, indices, approach = approach)

            # Save adjusted images
            data_dir = f'dataset/cifar-{num_classes}-adjusted/cifar-{num_classes}-{percentage*100}%-removed'
            save_imagefolder_image(data_dir, target, new_image, idx, dataset)


def get_salience_based_adjusted_data(sample_loader, ks, percentages, num_classes = 10, dataset = "train"):
    """
        Creates adjusted images based on different K's, and saves them.
        
        sample_loader: created dataloader, used to sample the images.
        ks           : amount of pixels which are adjusted within the image.
        percentages  : the percentages of pixels which are adjusted, used to save images in correct
                       directories.
        dataset      : Used to define which set is used.  
    """

    # Creates data directories if needed.
    if not os.path.exists(f'dataset/cifar-{num_classes}-adjusted'):
        create_data_dirs(percentages, num_classes)

    # Loops over sample loader to creates per sample every adjusted image, and saves them.
    for idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        # Compute saliency maps for the input data.
        _, cam, _ = fullgrad.saliency(data)

        # Find most important pixels, replace and save adjusted image.
        create_adjusted_images_and_save(idx, data, cam, target, ks, percentages, num_classes, dataset)


if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_name', type=str, default="VGG-11", help="Name of the model when saved")
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of examples to process in a batch')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs until break')
    parser.add_argument('--load_model', type=str, default='', help='Give location of weights to load model')
    parser.add_argument('--save_epochs', type=int, default=1, help="save model after epochs")
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda'")
    parser.add_argument('--save_model', type=bool, default=True, help="If set to false the model wont be saved.")
    parser.add_argument('--data_dir', type=str, default=PATH + 'dataset', help="data dir for dataloader")
    parser.add_argument('--dataset_name', type=str, default='/cifar10-imagefolder', help= "Name of dataset contained in the data_dir")
    parser.add_argument('--checkpoint_path', type=str, default=PATH + 'saved-models/vgg-11', help="model saving dir.")
    parser.add_argument('--dataset', type=str, default='cifar10', help="Select cifar10 or cifar100 dataset")

    config = parser.parse_args()

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    config.checkpoint_path = os.path.join(config.checkpoint_path, '{model}-{epoch}-{type}.pth')

    device = torch.device(config.device)

    # Create dataloaders
    shuffle = True
    if config.num_classes == 10:
        transform = [CIFAR_10_TRANSFORM, CIFAR_10_TRANSFORM]
    else:
        transform = [CIFAR_100_TRANSFORM_TRAIN, CIFAR_100_TRANSFORM_TEST]

    train_loader = load_data(config.batch_size, transform[0], True, 2, config.data_dir, 
                                config.dataset_name, train=True, name=config.dataset)
    test_loader = load_data(config.batch_size, transform[1], False, 2, config.data_dir, 
                                config.dataset_name, train=False, name=config.dataset)

    
    # Get sample and total pixels 
    sample_img = next(iter(train_loader))[0]
    total_pixels = sample_img.shape[2] * sample_img.shape[3]

    # Calculate k's based on percentages and total pixels
    percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    Ks = [round((k * total_pixels)) for k in percentages]

    # Load or train model
    model = vgg11(pretrained=False, device=device, im_size = sample_img.shape, num_classes=config.num_classes, class_size=512).to(device)
    if os.path.exists('saved-models/VGG-11-71-best.pth'):
        print("The model will now be loaded.")
        print(True if device == 'cuda' else False)
        model.load_state_dict(torch.load('saved-models/VGG-11-71-best.pth', map_location=torch.device('cpu')), True if device == 'cuda' else False)
    else:
        # Train model on cifar-100
        print("The model will now be trained.")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

        train(model, criterion, optimizer, scheduler, train_loader, test_loader, device,
            config.checkpoint_path, config.model_name, config.epochs, config.save_epochs)

    # Initialize FullGrad objects
    fullgrad = FullGrad(model, im_size = sample_img.shape, device = device)

    # Create ROAR images
    print("The adjusted data will now be created.")
    get_salience_based_adjusted_data(train_loader, Ks, percentages, dataset = "train")
    get_salience_based_adjusted_data(test_loader, Ks, percentages, dataset = "test")
