# Visualisation of filters in the first layer by heatmap plot

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results-directory', type=str, default="results", help='Result Directory')
parser.add_argument('--experiment', type=str, default="conv", help='Experiment name')
parser.add_argument('--seed', type=int, required=True, help='Random seed used')
parser.add_argument('--trial', type=int, required=True, help='Which trial to plot')
parser.add_argument('--n-modules', type=int, default=10, help='Number of modules')
parser.add_argument('--layer', type=int, default=0, help='Which layer to visualize')
parser.add_argument('--kernel-size', type=int, default=3, help='Kernel (filter) size')
parser.add_argument('--lambdas', nargs='+', type=float, default=[0.0, 1.0], help='Which lambdas to plot')
args = parser.parse_args()

def plot_filters_heatmap():
    for k, Lambda in enumerate(args.lambdas):
        # Load the filters of the first layer
        weight = torch.load("../{}/{}/{}/trial_{}/{}/weights/early_stop_weights.pt".format(args.results_directory, args.experiment, args.seed, args.trial, Lambda), map_location='cpu')
        w = weight.get('layers.{}.weight'.format(args.layer)).numpy()
        w = (w - np.min(w)) / (np.max(w) - np.min(w)) # Normalize to [0, 1]

        # Get number of output channels in that layer
        n_output_channels = w.shape[0] // args.n_modules

        # Load the test error of the model
        test_results = np.load("../{}/{}/{}/trial_{}/{}/test_results.npy".format(args.results_directory, args.experiment, args.seed, args.trial, Lambda))
        test_error = test_results[0, 2]

        # Average filters through depth(input channels)
        w = np.mean(w, axis=1)

        # Create an empty image
        pic = np.zeros([args.kernel_size * n_output_channels, args.kernel_size * args.n_modules])
  
        # Draw heatmap images of the filters (row:output channels, col:modules)
        for i in range(n_output_channels):
            for j in range(args.n_modules):
                pic[i*args.kernel_size:(i+1)*args.kernel_size, j*args.kernel_size:(j+1)*args.kernel_size] = w[j * n_output_channels + i, :, :]
        plt.subplot(1, len(args.lambdas), k+1)
        im = plt.imshow(pic, cmap='hot')
        #plt.colorbar(im)
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, args.n_modules * args.kernel_size - 0.5, args.kernel_size))
        ax.set_yticks(np.arange(-0.5, n_output_channels * args.kernel_size - 0.5, args.kernel_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("Modules\n\nTest error {}%".format(round(test_error * 100, 2)))
        if k == 0:
            ax.set_ylabel("Output Channels")
        ax.grid(color='black', linestyle='-', linewidth=1)
        ax.set_title("Lambda " + str(Lambda))

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

if __name__ == "__main__":
    plot_filters_heatmap()
