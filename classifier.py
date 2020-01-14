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
from models.vgg import vgg11

def load_cifar_data(batch_size, data_dir, data_set='train'):
    """
        load_cifar_data prepares a dataloader for the cifar dataset.
    """
    print('==> Preparing data..')
    if data_set == 'train':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        return trainloader
    elif data_set == 'test':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        return testloader
    else:
        print('set does not exist, please enter train/test as data_set argument')
        return None

def parse_epoch(dataloader, model, optimizer, criterion, device, train=True):
    '''
        Training and evaluation are put together to avoid duplicated code.
    '''
    if train:
        model.train()
        print('training mode')
    else:
        model.eval()

    losses, total, correct = 0, 0, 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        if train:
            optimizer.zero_grad()    
            outputs = model(data)
            loss = criterion.forward(outputs, target)
            loss.backward()
            optimizer.step()

            losses += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            print('batch: %d | Loss: %.3f | Acc: %.3f' % (batch_idx, losses/(batch_idx+1), 100.*correct/total))
        else: 
            with torch.no_grad():
                outputs = model(data)
                loss = criterion.forward(outputs, target)
                
                losses += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                print('batch: %d | Loss: %.3f | Acc: %.3f' % (batch_idx, losses/(batch_idx+1), 100.*correct/total))
    return correct/total

def train(model, criterion, optimizer, scheduler, trainloader, testloader, device,
        checkpoint_path, model_name, save_epochs):
    '''
        This function trains the model that is passed in the first argument,
        using the arguments used afterwards.
    '''
    best_acc = 0.0
    for epoch in range(0, config.epochs):
        print(optimizer)
        parse_epoch(trainloader, model, optimizer, criterion, device)
        torch.cuda.empty_cache()
        scheduler.step()
        accuracy = parse_epoch(testloader, model, optimizer, criterion, device, train=False)
        
        if accuracy > best_acc:
            torch.save(model.state_dict(), checkpoint_path.format(model=model_name, epoch=epoch, type='best'))
            best_acc = accuracy
            continue

        if not epoch % save_epochs:
            torch.save(model.state_dict(), checkpoint_path.format(model=model_name, epoch=epoch, type='normal'))    
            
def eval(model, criterion, optimizer, trainloader, testloader, device,
            load_model, save_epochs):
    """
        This function loads the model weights from the load_model location.
        Afterwards it is run through 1 epoch of the test dataset to get the accuracy.
    """
    model.load_state_dict(torch.load(load_model), True if device == 'cuda' else False)
    parse_epoch(testloader, model, optimizer, criterion, device, train=False)

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
    parser.add_argument('--checkpoint_path', type=str, default=PATH + 'saved-models/', help="model saving dir.")

    config = parser.parse_args()

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    config.checkpoint_path = os.path.join(config.checkpoint_path, '{model}-{epoch}-{type}.pth')


    device = torch.device(config.device)

    model = vgg11(pretrained=False, num_classes=config.num_classes, class_size=512).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

    trainloader = load_cifar_data(config.batch_size, config.data_dir, data_set='train')
    testloader = load_cifar_data(config.batch_size, config.data_dir, data_set='test')

    # Train the model
    if config.load_model:
        eval(model, criterion, None, trainloader, testloader, device,
            config.load_model, config.save_epochs)        
    else:
        train(model, criterion, optimizer, scheduler, trainloader, testloader, device,
            config.checkpoint_path, config.model_name, config.save_epochs)