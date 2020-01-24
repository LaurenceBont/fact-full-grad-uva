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
from classifier import train, eval
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from misc_functions import *
import csv

csv_dir = 'dataset/PPB-2017/PPB-2017-metadata.csv'
metadata = []
with open(csv_dir, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        metadata.append(row)


def racial_acc(dataloader, model, optimizer, criterion, device, train=True):
    '''
        Training and evaluation are put together to avoid duplicated code.
    '''
    model.eval()

    total = {'Male': {'lighter' : 0, 'darker': 0}, 'Female':  {'lighter' : 0, 'darker': 0} }
    correct = {'Male': {'lighter' : 0, 'darker': 0}, 'Female':  {'lighter' : 0, 'darker': 0} }

    image_index = 1

    losses = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        # print(batch_idx)
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            outputs = model(data)
            loss = criterion.forward(outputs, target)
            _, predicted = outputs.max(1)

            for i in range(target.size(0)):
                gender = metadata[image_index][2]
                tint = metadata[image_index][4]
                total[gender][tint] += 1
                correct[gender][tint] += predicted[i].eq(target[i]).sum().item()

                image_index += 1

    
    
    print("Male, lighter, accuracy :", correct['Male']['lighter']/total['Male']['lighter'])
    print("Male, darker, accuracy :", correct['Male']['darker']/total['Male']['darker'])
    print("Female, lighter, accuracy :", correct['Female']['lighter']/total['Female']['lighter'])
    print("Female, darker, accuracy :", correct['Female']['darker']/total['Female']['darker'])

def compute_save_fullgrad_saliency(sample_loader, unnormalize, save_path, device, fullgrad):
    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        # Compute saliency maps for the input data
        _, cam, _ = fullgrad.saliency(data)
        
        # Save saliency maps
        for i in range(data.size(0)):
            filename = save_path + str( (batch_idx+1) * (i+1)) 
            filename_simple = filename + '_simple'

            image = unnormalize(data[i,:,:,:].cpu())
            save_saliency_map(image, cam[i,:,:,:], filename + '.jpg')
           
if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    save_path = PATH + 'results/'
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_name', type=str, default="VGG-11", help="Name of the model when saved")
    parser.add_argument('--num_classes', type=int, default=2, help='Dimensionality of output sequence')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
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
    parser.add_argument('--save_saliency', type=bool, default=False, help="Set true if you want to save salienct")

    config = parser.parse_args()

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    config.checkpoint_path = os.path.join(config.checkpoint_path, '{model}-{epoch}-{type}.pth')


    device = torch.device(config.device)
    model = resnet50(num_classes=2).to(device)

    if config.load_model:
        fullgrad = FullGrad(model, im_size=(1,3,64,64))
        simple_fullgrad = SimpleFullGrad(model)
        model.load_state_dict(torch.load(config.load_model), True if device == 'cuda' else False)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

    num_workers = 1
    test_dir = PATH + 'dataset/extra_experiment/test'
    train_dir = PATH + 'dataset/extra_experiment/train'

    transform = transforms.Compose(
        [transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    unnormalize = NormalizeInverse(mean = [0.5, 0.5, 0.5],
                        std = [0.5, 0.5, 0.5])


    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=num_workers, shuffle=True)

    dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=num_workers)

    dataset = datasets.ImageFolder(root=PATH + 'dataset/saliency/', transform=transform)
    saliencyloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers)


    if config.save_saliency:
        compute_saliency_and_save(saliencyloader)
    

    racial_acc(testloader, model, optimizer, criterion, device)
