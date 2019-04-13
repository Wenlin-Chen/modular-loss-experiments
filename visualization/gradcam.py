"""
Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
https://arxiv.org/abs/1610.02391

This code is based on the implementation from Utku Ozbulak - github.com/utkuozbulak
"""

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

from gradcam_misc import get_example_params, save_class_activation_images, get_args


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_module, target_layer, n_modules):
        self.model = model
        self.target_module = target_module
        self.target_layer = target_layer
        self.gradients = None
        self.n_oc = None  # Number of out channels per module
        self.start = None  # Channel that the target module starts from
        self.end = None  # Channel that the target module ends
        self.n_modules = n_modules

    def save_gradient(self, grad):
        self.gradients = grad[:, self.start:self.end, :, :]

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        x = x.repeat(1, self.n_modules, 1, 1)
        for layer_pos, layer in enumerate(self.model.layers):
            self.n_oc = layer.out_channels // self.n_modules 
            x = F.relu(layer(x))  # Forward
            if layer_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                self.start = self.n_oc * self.target_module
                self.end = self.n_oc*(self.target_module+1)
                conv_output = x[:, self.start:self.end, :, :]  # Save the convolution output on that layer on that module
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        kernel = x.shape[2:]
        pool = nn.AvgPool2d(kernel)
        x = pool(x)

        # Forward pass on the classifier
        x = x.view(1, self.n_modules, -1).permute(1, 0, 2)
        x = self.model.fc(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_module, target_layer, n_modules):
        self.model = model
        self.target_module = target_module
        self.model.eval()

        # Define extractor
        self.extractor = CamExtractor(self.model, target_module, target_layer, n_modules)

    def generate_cam(self, input_image, target_class):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model
        conv_output, model_output = self.extractor.forward_pass(input_image)
        module_output = model_output[self.target_module]
        module_prob = F.softmax(module_output, dim=1)
        module_prediction = torch.argmax(module_prob).numpy()
        print("predicted class by the target module: " + str(module_prediction))

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Zero grads
        self.model.zero_grad()

        # Backward pass with specified target
        module_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]

        # Get convolution outputs
        target = conv_output.data.numpy()[0]

        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)

        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS)) # Resize to the shape of input image
        return cam


if __name__ == '__main__':
    # Get params
    args = get_args()
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(args.example_path, args.target_class, args.Lambda, args.results_directory, args.experiment, args.seed, args.trial)

    # Grad cam
    grad_cam = GradCam(pretrained_model, args.target_module, args.target_layer, args.n_modules)

    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, args.target_class)

    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)

    print('Grad cam completed')
