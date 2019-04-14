# Computing the displacements and distances travelled in parameter space

import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results-directory', type=str, default="results", help='Result Directory')
parser.add_argument('--experiment', type=str, default="conv", help='Experiment name')
parser.add_argument('--seed', type=int, required=True, help='Random seed used')
parser.add_argument('--trial', type=int, required=True, help='Which trial to plot')
parser.add_argument('--n-modules', type=int, default=10, help='Number of modules')
parser.add_argument('--layer', type=int, default=0, help='Which layer to compute')
parser.add_argument('--kernel-size', type=int, default=3, help='Kernel (filter) size')
parser.add_argument('--lambdas', nargs='+', type=float, default=[0.0, 1.0], help='Which lambdas to plot')
parser.add_argument('--learning-rates', nargs='+', type=float, default=[0.06, 0.02], help='Learning rate used corresponding to each Lambda')
parser.add_argument('--early-stop-epochs', nargs='+', type=int, required=True, help='Early stop at which epochs')
parser.add_argument('--bias', default=False, action='store_true', help='Whether to compute for bias')

args = parser.parse_args()

def get_initial_weights(bias=False):
    initial_weights = torch.load("../{}/{}/{}/trial_{}/initial_weights.pt".format(args.results_directory, args.experiment, args.seed, args.trial), map_location='cpu')
    initial_w = initial_weights.get('layers.{}.weight'.format(args.layer)).numpy()
    if bias:
        initial_b = initial_weights.get('layers.{}.bias'.format(args.layer)).numpy()
    else:
        initial_b = None
 
    return initial_w, initial_b

def get_grad_norms(Lambda, bias=False):
    grad_norms = np.load("../{}/{}/{}/trial_{}/{}/gradient_norms.npy".format(args.results_directory, args.experiment, args.seed, args.trial, Lambda))
    w_grad_norms = grad_norms[:, :, :, args.layer, 0]
    if bias:
        b_grad_norms = grad_norms[:, :, :, args.layer, 1]
    else:
        b_grad_norms = None

    return w_grad_norms, b_grad_norms

def compute_distances_between_start_end(initial_weights, Lambda, learning_rate, early_stop_epoch, bias=False):
    # Load the early stop weights
    weights = torch.load("../{}/{}/{}/trial_{}/{}/weights/early_stop_weights.pt".format(args.results_directory, args.experiment, args.seed, args.trial, Lambda), map_location='cpu')
    w = weights.get('layers.{}.weight'.format(args.layer)).numpy()
    if bias:
        b = weights.get('layers.{}.bias'.format(args.layer)).numpy()

    # Get the gradient norms
    grad_norms = get_grad_norms(Lambda, bias)

    # Get number of output channels in that layer
    n_output_channels = w.shape[0] // args.n_modules

    # Compute the displacements between initial and early stop weights
    w_displacement = initial_weights[0] - w
    if bias:
        b_displacement = initial_weights[1] - b

    # Compute the 2-norm of displacements and distances travelled of the weights of the ensemble
    w_displacement_norm = np.linalg.norm(w_displacement, ord=None)
    w_distance_travelled_norm = np.sum(learning_rate * grad_norms[0][:early_stop_epoch])
    if bias:
        b_displacement_norm = np.linalg.norm(b_displacement, ord=None)
        b_distance_travelled_norm = np.sum(learning_rate * grad_norms[1][:early_stop_epoch])
        print("Lambda {}, Ensemble: w displacement {}, b displacement {}, w distance travelled {}, b distance travelled {}".format(Lambda, w_displacement_norm, b_displacement_norm, w_distance_travelled_norm, b_distance_travelled_norm))
    else:
        print("Lambda {}, Ensemble: w displacement {}, w distance travelled {}".format(Lambda, w_displacement_norm, w_distance_travelled_norm))
      
    # Compute and print the 2-norm of displacements and distances travelled of the weights in each module
    for i in range(args.n_modules):
        start = i * n_output_channels
        end = (i + 1) * n_output_channels
        w_displacement_norm = np.linalg.norm(w_displacement[start:end], ord=None)
        w_distance_travelled_norm = np.sum(learning_rate * grad_norms[0][:early_stop_epoch, :, i])
        if bias:
            b_displacement_norm = np.linalg.norm(b_displacement[start:end], ord=None)
            b_distance_travelled_norm = np.sum(learning_rate * grad_norms[1][:early_stop_epoch, :, i])
            print("Lambda {}, Module {}: w displacement {}, b displacement {}, w distance travelled {}, b distance travelled {}".format(Lambda, i, w_displacement_norm, b_displacement_norm, w_distance_travelled_norm, b_distance_travelled_norm))
        else:
            print("Lambda {}, Module {}: w displacement {}, w distance travelled {}".format(Lambda, i, w_displacement_norm, w_distance_travelled_norm))
    
    print()

if __name__ == "__main__":
    initial_weights = get_initial_weights(bias=args.bias)
    for k, Lambda in enumerate(args.lambdas):
        compute_distances_between_start_end(initial_weights, Lambda, args.learning_rates[k], args.early_stop_epochs[k]+1, bias=args.bias)
