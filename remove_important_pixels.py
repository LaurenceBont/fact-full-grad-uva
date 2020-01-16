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

def return_k_index_argsort(salience_map, k, method):
    idx = np.argsort(salience_map.ravel())
    if method == "roar":
        return np.column_stack(np.unravel_index(idx[:-k-1:-1], salience_map.shape))
    elif method == "pp":
        return np.column_stack(np.unravel_index(idx[::-1][:k], salience_map.shape))
    elif method == "random":
        idx = np.random.choice(idx.shape[0], k, replace=False)
        return np.column_stack(np.unravel_index(idx, salience_map.shape))

def get_k_based_percentage(img, percentage):
    w, h = img.shape
    numb_pix = w*h
    return numb_pix * percentage

def calc_rgb_means(img):
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

def compute_saliency_and_save(k, method):
    former_outputs, new_images, image_counter = [], [], 0

    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        # Compute saliency maps for the input data.
        cam, model_output = fullgrad.saliency(data)

        # Find most important pixels and replace.
        for i in range(data.size(0)):
            # Append former output to a tensor.
            former_outputs.append(model_output[i])   

            # Get unnormalized image and heat map.
            sal_map = cam[i,:,:,:].squeeze()
            image = unnormalize(data[i,:,:,:])

            # Get k indices and replace within image
            indices = return_k_index_argsort(sal_map.detach().numpy(), k, method)
            new_image = replace_pixels(image, indices, 'zero')
            new_images.append(new_image)

            # Save adjusted images, if needed.
            if method == "roar":
                utils.save_image(new_image, f'pixels_removed/{method}/removal{k*100}%/img_id={image_counter}removal={k*100}%.jpeg')

            image_counter += 1
    image_counter = 0

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
        total += diff_den/(new_output[max_index]+former_output[max_index])

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

        
        




