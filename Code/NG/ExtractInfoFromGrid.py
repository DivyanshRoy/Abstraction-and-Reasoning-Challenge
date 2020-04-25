import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
import os
from os import listdir
from os.path import isfile, join
from enum import Enum
from importnb import Notebook
import matplotlib
import importlib

globalObjectId = 0

def removeColorAspect(grid):
    grid_np = np.array(grid)
    grid_np[grid_np != 0] = 9
    return grid_np.tolist()

def exploreObject(grid, H, W, r, c, heatmap, objectId):
    heatmap[r][c] = objectId
    poss_rows = [r]
    if (r+1 < H):
        poss_rows.append(r+1)
    if (r-1 >= 0):
        poss_rows.append(r-1)
    poss_cols = [c]
    if (c+1 < W):
        poss_cols.append(c+1)
    if (c-1 >= 0):
        poss_cols.append(c-1)
    
    for row in poss_rows:
        for col in poss_cols:
            if (row == r and col == c):
                continue
            elif (heatmap[row][col] != 0):
                continue
            elif (grid[row][col] == 0):
                continue
            exploreObject(grid, H, W, row, col, heatmap, objectId)
    return
        
    
def findHeatmap(grid):
    height = len(grid)
    width = len(grid[0])
    objectId = 1
    heatmap = [[0]*width for _ in range(height)]
    for h in np.arange(height):
        for w in np.arange(width):
            if grid[h][w] == 0 or heatmap[h][w] != 0:
                continue
            else:
                exploreObject(grid, height, width, h, w, heatmap, objectId)
                objectId += 1
    
    return heatmap

def isIdenticalObjects (obj1, obj2):
    (h1, w1) = obj1.shape
    (h2, w2) = obj2.shape
    if ((h1 == h2 and w1 == w2) or (h1 == w2 and w1 == h2)):
        if (np.array_equal(obj1, obj2)):
            return True
        elif (np.array_equal(obj1, np.rot90(obj2, 1))):
            return True
        elif (np.array_equal(obj1, np.rot90(obj2, 2))):
            return True
        elif (np.array_equal(obj1, np.rot90(obj2, 3))):
            return True
    
    return False
 
def isSimilarObjects (obj1, obj2):
    #Objects which are same in outline or shape

    def drawOutline (obj):        
        o1 = np.copy(obj, int)
        (h1, w1) = o1.shape

        for r in np.arange(h1):
            for c in np.arange(w1):
                if obj[r][c] == 0:
                    continue
                poss_rows = [r]
                if (r - 1 >= 0):
                    poss_rows.append(r-1)
                if (r + 1 < h1):
                    poss_rows.append(r+1)
                poss_cols = [c]
                if (c - 1 >= 0):
                    poss_cols.append(c-1)
                if (c + 1 < w1):
                    poss_cols.append(c+1)

                count = 0
                for row in poss_rows:
                    for col in poss_cols:
                        if obj[row][col] != 0 and not (row == r and col == c):
                            count += 1

                if (count == 8):
                    o1[r][c] = 0
        
        return o1
    
    o1 = drawOutline(obj1)
    o2 = drawOutline(obj2)
    return isIdenticalObjects(o1, o2)

def isCongruentObjects (obj1, obj2):
    return False

def isScaledObjects (obj1, obj2):
    
    def check (scaleH, scaleW, small, big):
        obj = np.zeros((small.shape[0] * scaleH, small.shape[1] * scaleW))
        idxs = np.where(small != 0)
        for r in idxs[0]:
            for c in idxs[1]:
                val = np.ones((scaleH, scaleW)) * small[r][c]
                obj[r * scaleH: (r+1)*scaleH, c * scaleW: (c+1)*scaleW] = val
        return isIdenticalObjects(obj, big)
    
    (h1, w1) = obj1.shape
    (h2, w2) = obj2.shape
    
    if (h1 % h2 == 0 and w1 % w2 == 0):
        return check(h1//h2, w1//w2, obj2, obj1)
    elif (h1 % w2 == 0 and w1 % h2 == 0):
        return check(w1//h2, h1//w2, obj2, obj1)
    elif (h2 % h1 == 0 and w2 % w1 == 0):
        return check(h2//h1, w2//w1, obj1, obj2)
    elif (h2 % w1 == 0 and w2 % h1 == 0):
        return check(w2//h1, h2//w1, obj1, obj2)
    else:
        return False

def isPartialObjects (obj1, obj2):
    def isPresent (small, big):
        if (small.shape[0] <= big.shape[0] and small.shape[1] <= big.shape[1]):
            (sh, sw) = small.shape
            (bh, bw) = big.shape
            for r in np.arange(bh - sh + 1):
                for c in np.arange(bw - sw + 1):
                    if (np.all(np.logical_not(np.logical_xor(small, big[r:r+sh, c:c+sw])))):
                        return True
        
        return False
    
    (h1, w1) = obj1.shape
    (h2, w2) = obj2.shape
    if (h1 >= h2 and w1 >= w2):
        if (isPresent(obj2, obj1)):
            return True, 1
        elif (isPresent(np.rot90(obj2, 2), obj1)):
            return True, 1        
    if (h1 >= w2 and w1 >= h2):
        if (isPresent(np.rot90(obj2, 1), obj1)):
            return True, 1
        elif (isPresent(np.rot90(obj2, 3), obj1)):
            return True, 1 
   
    if (h2 >= h1 and w2 >= w1):        
        if (isPresent(obj1, obj2)):
            return True, 2
        elif (isPresent(np.rot90(obj1, 2), obj2)):
            return True, 2   
    if (h2 >= h1 and w2 >= w1):
        if (isPresent(np.rot90(obj1, 1), obj2)):
            return True, 2
        elif (isPresent(np.rot90(obj1, 3), obj2)):
            return True, 2  
    
    return False, 0
    
def findObjectsFromHeatmap (grid, heatmap):
    global globalObjectId
    heatmap_np = np.array(heatmap)
    grid_np = np.array(grid)
    maxObjects = np.max(heatmap_np)
    objects = []
    for obj in np.arange(1, maxObjects+1):
        idxs = np.where(heatmap == obj)
        min_r, min_c, max_r, max_c = np.min(idxs[0]), np.min(idxs[1]), np.max(idxs[0]), np.max(idxs[1])
        mask = np.zeros(heatmap_np.shape)
        mask[heatmap_np == obj] = 1
        structure = (grid_np * mask)[min_r:max_r+1, min_c:max_c+1]
        on_cells = np.count_nonzero(structure)
        
        objects.append({"idx": obj-1, 
                        "global_idx": globalObjectId,
                        "structure": structure, 
                        "bounds": (min_r, min_c, max_r, max_c), 
                        "properties": [],
                        "similar_objects": [],
                        "partial_objects": [],
                        "identical_objects": [],
                        "scaled_objects": [],
                        "ON_cells": on_cells
                       })
        globalObjectId += 1
    
    for idx in np.arange(len(objects)):
        obj = objects[idx]['structure']
        for idx1 in np.arange(idx+1, len(objects)):
            obj1 = objects[idx1]['structure']
            if (isIdenticalObjects(obj, obj1)):
                objects[idx]['identical_objects'].append(idx1)
                objects[idx1]['identical_objects'].append(idx)
            if (isSimilarObjects(obj, obj1)):
                objects[idx]['similar_objects'].append(idx1)
                objects[idx1]['similar_objects'].append(idx)
            if (isScaledObjects(obj, obj1)):
                objects[idx]['scaled_objects'].append(idx1)
                objects[idx1]['scaled_objects'].append(idx)
            
            res, temp = isPartialObjects(obj, obj1)
            if (res):
                if (temp == 1):
                    objects[idx]['partial_objects'].append(idx1)
                elif (temp == 2):
                    objects[idx1]['partial_objects'].append(idx)
                else:
                    assert False, "temp cannot be 0 if partial object found"
                
        
    return objects
        
def findCharacteristics(grid):
    characters = {}
    #Find objects
    objects = findObjectsFromHeatmap(grid, findHeatmap(grid))
    characters['objects'] = objects
    characters['distinct_objects'] = []
    
    return characters

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
    
def show_single_pair_training_example_visually (input_grid, output_grid, label, drawGridAroundObjects):
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
    if (drawGridAroundObjects):
        drawGridCharacteristics(input_grid, ax[0])
    
    
    ax[1].imshow(output_grid, cmap=cmap, norm = norm)
    ax[1].set_xticks(np.arange(widthO) + 0.5)
    ax[1].set_yticks(np.arange(heightO) + 0.5)
    ax[1].grid(linewidth=1.5)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_title(label + 'Output')
    if (drawGridAroundObjects):
        drawGridCharacteristics(output_grid, ax[1])
    plt.show()
    plt.close(fig)
    
def show_single_training_example_visually (train_list, test_list, filename, removeColorAsp, drawGridAroundObjects):
    #plt.close()
    basename = os.path.basename(filename)
    for pairIdx in np.arange(len(train_list)):
        input_grid = train_list[pairIdx]['input']
        output_grid = train_list[pairIdx]['output']
        if (removeColorAsp):
            input_grid, output_grid = removeColorAspect(input_grid), removeColorAspect(output_grid)
        show_single_pair_training_example_visually(input_grid, output_grid, basename + ' Tr ', drawGridAroundObjects)

    for pairIdx in np.arange(len(test_list)):
        input_grid = test_list[pairIdx]['input']
        output_grid = test_list[pairIdx]['output']
        if (removeColorAsp):
            input_grid, output_grid = removeColorAspect(input_grid), removeColorAspect(output_grid)
        show_single_pair_training_example_visually(input_grid, output_grid, basename + ' Te ', drawGridAroundObjects)
        
class Characteristics():
    def __init__(self, train_list, removeColorAsp):
        global globalObjectId
        #Inputs:
        #train_list: list of dicts {"input":[[]], "output":[[]]}
        #removeColorAsp: Whether to remove color aspect from dataset or not
        
        globalObjectId = 0
        #Gather individual grid stats
        self.removeColorAsp = removeColorAsp
        
        #Gather colored/non colored data
        self.train_list = [] 
        for pairIdx in np.arange(len(train_list)):
            input_grid = train_list[pairIdx]['input']
            output_grid = train_list[pairIdx]['output']
            if (self.removeColorAsp):
                input_grid, output_grid = removeColorAspect(input_grid), removeColorAspect(output_grid)
            self.train_list.append({'input': input_grid, 'output': output_grid})
        
        #Start finding characteristics
        self.train_individual_char = []
        self.global_chars = {}
        self.global_chars['objects'] = []
        for pairIdx in np.arange(len(self.train_list)):
            input_grid = self.train_list[pairIdx]['input']
            output_grid = self.train_list[pairIdx]['output']
            input_chars = findCharacteristics(input_grid)
            output_chars = findCharacteristics(output_grid)
            char_pair = {'input':input_chars, 'output':output_chars}
            self.train_individual_char.append(char_pair) 
            self.global_chars['objects'] += input_chars['objects'] + output_chars['objects']
            
        
          
    def printChars(self):
        for pairIdx in np.arange(len(self.train_individual_char)):
            print('<========= Input ==========>', pairIdx)
            print(self.train_individual_char[pairIdx]['input'])
            
            print('<========= Output ==========>', pairIdx)
            print(self.train_individual_char[pairIdx]['output'])            
            