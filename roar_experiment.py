#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder 
    and dump them in a results folder """

import torch

from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import *
from models.resnet import *
from misc_functions import *
from utils import prepare_cifar_data, load_cifar_data, CIFAR_100_TRANSFORM_TRAIN, CIFAR_100_TRANSFORM_TEST




def experiment(dataset, experiments = [10, 33, 66, 99]):
    model = train_model(dataset, model_type)

    accuracy_list = []
    for experiment in experiments:
        adjusted_data = get_adjusted_data(model, salience_method, experiment)
        model_k_data = train_model(adjusted_data, model_type)
        accuracy_list.append(parse_epoch(adjusted_data, model_k_data, None, criterion, device, train=False)))


# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

batch_size = 4
total_pixels = 244*244

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Dataset loader for sample images
sample_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(dataset, transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                   ])),
    batch_size= batch_size, shuffle=False)

transform_image = transforms.ToPILImage()
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])


# uncomment to use VGG
model = vgg16_bn(pretrained=True).to(device)
# model = resnet18(pretrained=True).to(device)

# Initialize FullGrad objects
fullgrad = FullGrad(model)
simple_fullgrad = SimpleFullGrad(model)

save_path = PATH + 'results/'




def pixel_pertubation():
    percentages = [0.001, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1]
    Ks = [round((k * total_pixels)) for k in percentages]
    
    full_grad_results, random_results = [], []    
    for k_index, k in enumerate(Ks):
        faoc_full_grad = compute_pertubation(k, method = "pp") # fractional absolute output change
        # faoc_random = ...

        full_grad_results.append(faoc_full_grad)
        print("----------------------------------")
        print(f'petje for {percentages[k_index]*100}% = { faoc_full_grad }')

    plt.plot(percentages, full_grad_results, marker = 'o')
    # plt.plot(percentages, random_results, marker = 'o')
    plt.show()

if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_name', type=str, default="VGG-11", help="Name of the model when saved")
    parser.add_argument('--num_classes', type=int, default=100, help='Dimensionality of output sequence')
    parser.add_argument('--batch_size', type=int, default=425, help='Number of examples to process in a batch')
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


    device = torch.device(config.device)

    model = vgg11(pretrained=False, num_classes=config.num_classes, class_size=512).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    trainloader = load_cifar_data(config.batch_size, CIFAR_100_TRANSFORM_TRAIN,
                                True, 2, config.data_dir, config.dataset_name, train=True)
    testloader = load_cifar_data(config.batch_size, CIFAR_100_TRANSFORM_TEST,
                                False, 2, config.data_dir, config.dataset_name, train=False)

    # Train the model
    if config.load_model:
        eval(model, criterion, None, trainloader, testloader, device,
            config.load_model, config.save_epochs)        
    else:
        train(model, criterion, optimizer, trainloader, testloader, device,
            config.checkpoint_path, config.model_name, config.save_epochs)

        
        




