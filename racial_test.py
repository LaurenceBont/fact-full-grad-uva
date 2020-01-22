#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#
""" 
This document contains the functions to classify the images,
with a certain dataLoader. Calling this file as main function
it allows for certain flags to be set and instantly run the classifier
and save it
"""
import torch
from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from models.vgg import vgg11, vgg16, vgg19
from models.resnet import resnet50
from utils import prepare_data, load_data, CIFAR_100_TRANSFORM_TRAIN, CIFAR_100_TRANSFORM_TEST, CIFAR_10_TRANSFORM, load_imageFolder_data
from classifier import train

if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_name', type=str, default="VGG-11", help="Name of the model when saved")
    parser.add_argument('--num_classes', type=int, default=2, help='Dimensionality of output sequence')
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of examples to process in a batch')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs until break')
    parser.add_argument('--load_model', type=str, default='', help='Give location of weights to load model')
    parser.add_argument('--save_epochs', type=int, default=1, help="save model after epochs")
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda'")
    parser.add_argument('--save_model', type=bool, default=True, help="If set to false the model wont be saved.")
    parser.add_argument('--data_dir', type=str, default=PATH + 'dataset', help="data dir for dataloader")
    parser.add_argument('--dataset_name', type=str, default='/extra_experiment', help= "Name of dataset contained in the data_dir")
    parser.add_argument('--checkpoint_path', type=str, default=PATH + 'saved-models/', help="model saving dir.")
    parser.add_argument('--dataset', type=str, default='cifar10', help="Select cifar10 or cifar100 dataset")

    config = parser.parse_args()

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    config.checkpoint_path = os.path.join(config.checkpoint_path, '{model}-{epoch}-{type}.pth')


    device = torch.device(config.device)

    # model = vgg16(pretrained=False, device=device, num_classes=config.num_classes, class_size=512 * 2 * 2).to(device)

    model = resnet50(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

    num_workers = 1
    test_dir = PATH + 'dataset/extra_experiment/test'
    train_dir = PATH + 'dataset/extra_experiment/train'

    transform = transforms.Compose(
        [transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=num_workers, shuffle=True)

    dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=num_workers)

    print(len(trainloader))

    train(model, criterion, optimizer, scheduler, trainloader, testloader, device,
        config.checkpoint_path, config.model_name, config.save_epochs, epochs=config.epochs)