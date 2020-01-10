#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" This document classifies images """

import torch
from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.optim as optim
import os

from models.vgg import vgg11

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_data(batch_size, data_set='train'):
    print('==> Preparing data..')
    if data_set == 'train':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR100(root=dataset, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        return trainloader
    elif data_set == 'test':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR100(root=dataset, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        return testloader
    else:
        print('set does not exist, please enter train/test as data_set argument')
        return None

def train(dataloader, model, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion.forward(outputs, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        print(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(dataloader, model, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return correct/total


if __name__ == "__main__":
    model = vgg11(pretrained=False, num_classes=100).to(device)
    lr = 0.1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainloader = load_data(425, data_set='train')
    testloader = load_data(1000, data_set='test')

    while True:
        train(trainloader, model, optimizer, criterion)
        torch.cuda.empty_cache()
        if test(testloader, model, criterion) > 0.60:
            break
        




