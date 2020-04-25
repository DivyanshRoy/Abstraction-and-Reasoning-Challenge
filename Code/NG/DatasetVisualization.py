import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
import os
from os import listdir
from os.path import isfile, join
from enum import Enum
import ExtractInfoFromGrid
import importlib
importlib.reload(ExtractInfoFromGrid)
from ExtractInfoFromGrid import *
import matplotlib

def read_single_training_example (filename):
    #print("Reading data from " + filename)
    with open(filename) as file:
        data = json.load(file)
    train_list = data['train']
    test_list = data['test']        
    return train_list, test_list

def drawGridCharacteristics(grid, axs):
    H = len(grid)
    W = len(grid[0])
    chars = findCharacteristics(grid)
    objs = chars['objects']
    buffer = 0.1
    for obj in objs:
        (min_r, min_c, max_r, max_c) = obj['bounds']
        axs.add_patch(matplotlib.patches.Rectangle((min_c - 0.5 + buffer, min_r - 0.5 + buffer), max_c - min_c + 1 - 2 * buffer, max_r - min_r + 1 - 2 * buffer, fill = False, color = 'green', linewidth=3.0))
    return
    
def show_single_pair_training_example_visually (input_grid, output_grid, label):
    heightI = len(input_grid)
    widthI = len(input_grid[0])
    heightO = len(output_grid)
    widthO = len(output_grid[0])    
    cmap = mpl.colors.ListedColormap(['black', 'purple', 'red', 'green', 'yellow', 'gray', 'fuchsia', 'orange', 'cyan', 'brown'])
    bounds = np.arange(11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 2, figsize=(max(widthO, widthI), max(heightO, heightI)))
    ax[0].imshow(input_grid, cmap=cmap, norm = norm)
    ax[0].set_xticks(np.arange(widthI) + 0.5)
    ax[0].set_yticks(np.arange(heightI) + 0.5)
    ax[0].grid(linewidth=1.5)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_title(label + 'Input')
    drawGridCharacteristics(input_grid, ax[0])
    
    
    ax[1].imshow(output_grid, cmap=cmap, norm = norm)
    ax[1].set_xticks(np.arange(widthO) + 0.5)
    ax[1].set_yticks(np.arange(heightO) + 0.5)
    ax[1].grid(linewidth=1.5)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_title(label + 'Output')
    drawGridCharacteristics(output_grid, ax[1])
    plt.show()
    plt.close(fig)

def removeColorAspect(grid):
    grid_np = np.array(grid)
    grid_np[grid_np != 0] = 9
    return grid_np.tolist()
    
def show_single_training_example_visually (train_list, test_list, filename, removeColorAsp = False):
    #plt.close()
    basename = os.path.basename(filename)
    for pairIdx in np.arange(len(train_list)):
        input_grid = train_list[pairIdx]['input']
        output_grid = train_list[pairIdx]['output']
        if (removeColorAsp):
            input_grid, output_grid = removeColorAspect(input_grid), removeColorAspect(output_grid)
        show_single_pair_training_example_visually(input_grid, output_grid, basename + ' Tr ')

    for pairIdx in np.arange(len(test_list)):
        input_grid = test_list[pairIdx]['input']
        output_grid = test_list[pairIdx]['output']
        if (removeColorAsp):
            input_grid, output_grid = removeColorAspect(input_grid), removeColorAspect(output_grid)
        show_single_pair_training_example_visually(input_grid, output_grid, basename + ' Te ')    