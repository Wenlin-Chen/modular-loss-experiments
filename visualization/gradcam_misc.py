"""
Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
https://arxiv.org/abs/1610.02391

This code is based on the implementation from Utku Ozbulak - github.com/utkuozbulak
"""

import sys
sys.path.append('../src/')
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import models
import torch
import argparse
from torch.autograd import Variable


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('./output'):
        os.makedirs('./output')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()

    # Save image
    path_to_file = os.path.join('./output/' + file_name + '.jpg')
    save_image(gradient, path_to_file)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./output'):
        os.makedirs('./output')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'jet')

    # Save colored heatmap
    path_to_file = os.path.join('./output/' + file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)

    # Save heatmap on iamge
    path_to_file = os.path.join('./output/' + file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)

    # SAve grayscale heatmap
    path_to_file = os.path.join('./output/' + file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)

    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('LA').convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def save_image(im, path):
    """
        Saves a numpy matrix of shape D(1 or 3) x W x H as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image

    TODO: Streamline image saving, it is ugly.
    """
    if isinstance(im, np.ndarray):
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=0)
        if im.shape[0] == 1:
            # Converting an image with depth = 1 to depth = 3, repeating the same values
            # For some reason PIL complains when I want to save channel image as jpg without
            # additional format in the .save()
            im = np.repeat(im, 3, axis=0)

        # Convert to values to range 1-255 and W,H,D
        # A bandaid fix to an issue with gradcam
        if im.shape[0] == 3 and np.max(im) == 1:
            im = im.transpose(1, 2, 0) * 255
        elif im.shape[0] == 3 and np.max(im) > 1:
            im = im.transpose(1, 2, 0)
        im = Image.fromarray(im.astype(np.uint8))
    im = im.resize((224, 224))
    im.save(path)


def preprocess_image(pil_im):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Cifar 10)
    mean = [125.3/255.0, 123.0/255.0, 113.9/255.0]
    std = [63.0/255.0, 62.1/255.0, 66.7/255.0]
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H

    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]

    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()

    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)

    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)

    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-125.3/255.0, -123.0/255.0, -113.9/255.0]
    reverse_std = [255.0/63.0, 255.0/62.1, 255.0/66.7]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)
    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)

    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())

    return pos_saliency, neg_saliency


def get_model(Lambda, results_directory, experiment, seed, trial):
    input_shape = (3, 32, 32)
    output_dim = 10
    num_filters = [32, 64, 64, 128, 128]
    kernel_sizes = [3, 3, 3, 3, 3]
    strides = [1, 2, 1, 2, 1]
    dilations = [1, 1, 1, 1, 1]
    num_modules = 10
    activation = "relu"
    final_layer = "avg"
    
    dir_weights = "../{}/{}/{}/trial_{}/{}/weights/early_stop_weights.pt".format(results_directory, experiment, seed, trial, Lambda)

    model = models.ConvModel(input_shape, output_dim, num_filters, kernel_sizes, strides, dilations, num_modules, activation, final_layer)
    model.load_state_dict(torch.load(dir_weights, map_location='cpu')) 

    return model


def get_example_params(example_path, target_class, Lambda, results_directory, experiment, seed, trial):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    file_name_to_export = "output"
    # Read image
    original_image = Image.open(example_path).convert('RGB')
    # Resize the image to 224*224 and save it
    resized_im = original_image.resize((224, 224))
    if not os.path.exists('./output'):
        os.makedirs('./output')
    resized_im.save('./output/resized.png')
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    pretrained_model = get_model(Lambda, results_directory, experiment, seed, trial)
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example-path', type=str, required=True, help='Which example to visualize')
    parser.add_argument('--target-class', type=int, required=True, help='Target class in Grad CAM')
    parser.add_argument('--Lambda', type=float, required=True, help='Which Lambda to visualize')
    parser.add_argument('--target-module', type=int, required=True, help='Target module in Grad CAM')
    parser.add_argument('--target-layer', type=int, required=True, help='Which layer to visualize - always the last convolutional layer')
    
    parser.add_argument('--results-directory', type=str, default="results", help='Result Directory')
    parser.add_argument('--experiment', type=str, default="conv", help='Experiment name')
    parser.add_argument('--seed', type=int, required=True, help='Random seed used')
    parser.add_argument('--trial', type=int, required=True, help='Which trial to plot')
    parser.add_argument('--n-modules', type=int, default=10, help='Number of modules')    

    args = parser.parse_args()
    return args
