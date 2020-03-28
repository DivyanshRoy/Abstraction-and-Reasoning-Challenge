import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x

net = Net()

def model(data):
    output_shape = np.array(data['train'][0]['output']).shape
    # return data