# Each time we remove a module whose parameters have the smallest displacement

import sys
sys.path.append('../src/')
import torch
import argparse
import torchvision
import numpy as np
import torch.nn.functional as F
import models
from torchvision.transforms import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--results-directory', type=str, default="results", help='Result Directory')
parser.add_argument('--experiment', type=str, default="conv", help='Experiment name')
parser.add_argument('--seed', type=int, required=True, help='Random seed used')
parser.add_argument('--trial', type=int, required=True, help='Which trial to use')
parser.add_argument('--n-modules', type=int, default=10, help='Number of modules')
parser.add_argument('--Lambda', type=float, default=1.0, help='Which lambda to use')
parser.add_argument('--batch-size', type=int, default=100, help='Test batch size')
parser.add_argument('--drop-order', nargs='+', type=int, required=True, help='In which order the modules are dropped')
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

def test_with_some_modules_removed(model, testloader, keep):
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            score = model(images)
            score_bar = torch.mean(score[keep], dim=0)
            prob_bar = F.softmax(score_bar, dim=1)
            pred_bar = torch.argmax(prob_bar, dim=1)
            total += labels.size(0)
            correct += torch.sum(pred_bar == labels)
        acc = correct.data.cpu().numpy() / total
        print("With Modules {}, the test accuracy is {}".format(keep, acc))            
    
if __name__ == "__main__":
    model = get_model()
    testloader = get_testset()
    keep = list(range(args.n_modules))
    for i in range(args.n_modules):
        test_with_some_modules_removed(model, testloader, keep)
        if len(keep) <= 1:
            break
        else:
            keep.remove(args.drop_order[i])
