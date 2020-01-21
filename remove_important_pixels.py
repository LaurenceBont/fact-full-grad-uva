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
from utils import *

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
data_dir = PATH + 'dataset'

batch_size = 4
total_pixels = 244*244

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Dataset loader for sample images
"""
sample_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(data_dir, transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                   ])),
    batch_size= batch_size, shuffle=False)
"""

# Dataset loader for sample images
train_loader = load_cifar_data(1, CIFAR_100_TRANSFORM_TRAIN,
                                True, 2, data_dir, '/cifar-100-imageFolder', train=True)
test_loader = load_cifar_data(1, CIFAR_100_TRANSFORM_TEST,
                                False, 2, data_dir, '/cifar-100-imageFolder', train=False)


transform_image = transforms.ToPILImage()
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])


# uncomment to use VGG
print("The model will now be loaded.")
sample_img = next(iter(train_loader))[0]
model = vgg11(pretrained=False, im_size = sample_img.shape, num_classes=100, class_size=512).to(device)
model.load_state_dict(torch.load('saved-models/VGG-11-71-best.pth', map_location=torch.device('cpu')), True if device == 'cuda' else False)
# model = resnet18(pretrained=True).to(device)

# Initialize FullGrad objects
fullgrad = FullGrad(model)
simple_fullgrad = SimpleFullGrad(model)

save_path = PATH + 'results/'


def compute_saliency_and_save(k, method):
    former_outputs, new_images, image_counter = [], [], 0

    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        # Compute saliency maps for the input data.
        input_grad, full_grad, model_output = fullgrad.saliency(data)

        # Find most important pixels and replace.
        for i in range(data.size(0)):
            # Append former output to a tensor.
            former_outputs.append(model_output[i])   

            # Get unnormalized image and heat map.
            sal_map = full_grad[i,:,:,:].squeeze()
            image = unnormalize(data[i,:,:,:])

            # Get k indices and replace within image
            indices = return_k_index_argsort(sal_map.detach().numpy(), k, method)
            new_image = replace_pixels(image, indices, 'zero')
            new_images.append(new_image)

    return torch.stack(former_outputs), torch.stack(new_images)
    

def compute_pertubation(k, method = 'pp'):
    # Get adjusted images and fetch former outputs
    former_outputs, new_images = compute_saliency_and_save(k, method)

    # Normalise images again
    # new_images = transform_image(new_images)
    # normalized_images = normalize(new_images)

    # Create new outputs
    new_model_output = model.forward(new_images)

    # Calculate absolute fractional output change
    total = 0
    for i, former_output in enumerate(former_outputs):
        new_output = new_model_output[i]

        # Get argmax first outpout and calc difference
        max_index = former_output.argmax()
        diff_den = abs(new_output[max_index]-former_output[max_index])
        total += diff_den/(former_output[max_index])

    return (total/former_output.size(0))

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
    # Create folder to saliency maps
    create_folder(save_path)
    # compute_saliency_and_save()
    pixel_pertubation()
    print("Pertubation calculated")

        
        




