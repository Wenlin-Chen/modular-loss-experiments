# Visualisation of confusion matrix

import sys
sys.path.append('../src/')
import torch
import argparse
import torchvision
import numpy as np
import torch.nn.functional as F
import models
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--results-directory', type=str, default="results", help='Result Directory')
parser.add_argument('--experiment', type=str, default="conv", help='Experiment name')
parser.add_argument('--seed', type=int, required=True, help='Random seed used')
parser.add_argument('--trial', type=int, required=True, help='Which trial to plot')
parser.add_argument('--n-modules', type=int, default=10, help='Number of modules')
parser.add_argument('--Lambda', type=float, required=True, help='Which lambda to plot')
parser.add_argument('--batch-size', type=int, default=100, help='Test batch size')
args = parser.parse_args()

def get_model():
    input_shape = (3, 32, 32)
    output_dim = 10
    num_filters = [32, 64, 64, 128, 128]
    kernel_sizes = [3, 3, 3, 3, 3]
    strides = [1, 2, 1, 2, 1]
    dilations = [1, 1, 1, 1, 1]
    activation = "relu"
    final_layer = "avg"
    dir_weights = "../{}/{}/{}/trial_{}/{}/weights/early_stop_weights.pt".format(args.results_directory, args.experiment, args.seed, args.trial, args.Lambda)

    model = models.ConvModel(input_shape, output_dim, num_filters, kernel_sizes, strides, dilations, args.n_modules, activation, final_layer)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    model.load_state_dict(torch.load(dir_weights, map_location=device))    
    return model

def get_testset():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR10(root='../data/', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    return testloader

def get_confusion_matrices(model, testloader):
    conf = [[] for i in range(args.n_modules)]
    conf_bar = []

    with torch.no_grad():
        for images, labels in testloader:
            score = model(images)
            prob = F.softmax(score, dim=2)
            pred = torch.argmax(prob, dim=2)
            for m in range(args.n_modules):
                pairs = torch.stack([pred[m, :], labels], dim=1)
                conf[m].append(pairs)
            score_bar = torch.mean(score, dim=0)
            prob_bar = F.softmax(score_bar, dim=1)
            pred_bar = torch.argmax(prob_bar, dim=1)
            pairs = torch.stack([pred_bar, labels], dim=1)
            conf_bar.append(pairs)
    
    return conf, conf_bar

def plot_confusion_matrices(conf, conf_bar, n_classes=10, scatter_size=30, scatter_alpha=0.003):
    if args.n_modules % 5 == 0:
        n_rows = args.n_modules // 5 + 1
    else:
        n_rows = args.n_modules // 5

    fig = plt.figure()
    for m in range(args.n_modules):
        conf[m] = torch.cat(conf[m]).numpy()
        ax = fig.add_subplot(n_rows, 5, m+1)
        plt.scatter(conf[m][:, 0], conf[m][:, 1], s=scatter_size, alpha=scatter_alpha)
        plt.xlim(-1, n_classes)
        plt.ylim(-1, n_classes)
        ax.invert_yaxis()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("Module {}".format(m))

    conf_bar = torch.cat(conf_bar).numpy()
    ax = fig.add_subplot(n_rows, 5, args.n_modules+1)
    plt.scatter(conf_bar[:, 0], conf_bar[:, 1], s=scatter_size, alpha=scatter_alpha)
    plt.xlim(-1, n_classes)
    plt.ylim(-1, n_classes)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Ensemble prediction")

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

if __name__ == "__main__":
    model = get_model()
    testloader = get_testset()
    conf, conf_bar = get_confusion_matrices(model, testloader)
    plot_confusion_matrices(conf, conf_bar)
