#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder 
    and dump them in a results folder """

import torch

from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os
import cv2

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import *
from models.resnet import *
from misc_functions import *

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

batch_size = 4

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

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])


# uncomment to use VGG
# model = vgg16_bn(pretrained=True)
model = resnet18(pretrained=True).to(device)

# Initialize FullGrad objects
fullgrad = FullGrad(model)
simple_fullgrad = SimpleFullGrad(model)

save_path = PATH + 'results/'

# Found this function on stackoverflow 
# url: https://stackoverflow.com/questions/43386432/how-to-get-indexes-of-k-maximum-values-from-a-numpy-multidimensional-array
# Other approach: flatten --> max --> modulo row len om index te getten.
def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))

def get_k_based_percentage(img, percentage):
    w, h = img.shape
    numb_pix = w*h
    return numb_pix * percentage

def calc_mean_channels(img):
    mean_r = torch.mean(img[0,:,:])
    mean_g = torch.mean(img[1,:,:])
    mean_b = torch.mean(img[2,:,:])

    return mean_r, mean_g, mean_b

def replace_pixels(img, idx, approach = 'zero'):
    if approach == 'zero':
        for x,y in idx:
            img[:,x,y] = 0
    elif approach == 'mean':
        mean_r, mean_g, mean_b = calc_mean_channels(img)
        for x,y in idx:
            img[0,x,y] = mean_r
            img[1,x,y] = mean_g
            img[2,x,y] = mean_b

    return img

def compute_saliency_and_save():
    former_output, new_images, image_counter = [], [], 0

    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        # Compute saliency maps for the input data.
        cam, output_model = fullgrad.saliency(data)
        former_output.append(output_model)

        # Find most important pixels and replace.
        for i in range(data.size(0)):
            sal_map = cam[i,:,:,:].squeeze()
            image = unnormalize(data[i,:,:,:])

            max_indexes = k_largest_index_argsort(sal_map.detach().numpy(), k = 5018)
            new_image = replace_pixels(image, max_indexes, 'zero')
            new_images.append(new_image)

            # Unnormalize and save images with the found pixels changed.
            # new_image = unnormalize(new_image)
            utils.save_image(new_image, 'pixels_removed/' + str(image_counter) + '.jpeg')
            image_counter += 1
    image_counter = 0

    return former_output, new_images

def compute_pertubation():
    former_output, new_images = compute_saliency_and_save()

    # normalized_images = normalize(new_images)
    new_model_output = model.forward(np.asarray(new_images).from_numpy())

    # Calculate rare shit
    max_index = output_model.argmax()
    diff = abs(new_model_output[max_index]-output_model[max_index]).sum()
    print(diff)

            


if __name__ == "__main__":
    # Create folder to saliency maps
    create_folder(save_path)
    # compute_saliency_and_save()
    compute_pertubation()
    print('Saliency maps saved.')

        
        




