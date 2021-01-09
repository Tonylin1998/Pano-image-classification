"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

import sys

from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            print(grad_in[0])
            self.gradients = grad_in[1]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class, cnn_layer, filter_pos):
        self.model.zero_grad()
        # Forward pass
        x = input_image
        for index, layer in enumerate(self.model.features):
            print(index)
            # Forward pass layer by layer
            # x is not used after this point because it is only needed to trigger
            # the forward hook function
            x = layer(x)
            # Only need to forward until the selected layer is reached
            if index == cnn_layer:
                # (forward hook function triggered)
                break
        conv_output = torch.sum(torch.abs(x[0, filter_pos]))
        # Backward pass
        conv_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.ReLU(True),

                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(True)
            )

        self.class_classifier = nn.Sequential(
                nn.Linear(64*64*64, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout2d(),

                nn.Linear(1024, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(True),

                nn.Linear(100, 2),
            )

    def forward(self, x):
        x = self.features(x).view(-1, 64*64*64)
        #print(x.size())
        output = self.class_classifier(x)
        return output

if __name__ == '__main__':
    for filter_pos in range(256):
        cnn_layer = 2
        #filter_pos = 5
        target_example = 2  # Spider
        (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
            get_example_params(target_example)

        print(filter_pos)
        pretrained_model = ConvNet()
        pretrained_model.load_state_dict(torch.load(sys.argv[1], map_location=lambda storage, loc:storage))

        # File export name
        file_name_to_export = file_name_to_export + '_layer' + str(cnn_layer) + '_filter' + str(filter_pos)
        # Guided backprop
        GBP = GuidedBackprop(pretrained_model)
        # Get gradients
        guided_grads = GBP.generate_gradients(prep_img, target_class, cnn_layer, filter_pos)
        # Save colored gradients
        print(guided_grads.shape)
        save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
        # Convert to grayscale
        grayscale_guided_grads = convert_to_grayscale(guided_grads)
        # Save grayscale gradients
        print(grayscale_guided_grads.shape)
        save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
        # Positive and negative saliency maps
        #pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        #save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
        #save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
        print('Layer Guided backprop completed')
