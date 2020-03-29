import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from os import listdir
from os.path import isfile, join
matplotlib.use('TkAgg')
from model import model
import trainer
# task = "train"
task = "visualize"

path = "ARC-master/data/training/"

filenames = [f for f in listdir(path) if isfile(join(path, f))]
correct = 0
for filename in filenames:
    with open(path+"/"+filename) as json_file:
        data = json.load(json_file)
        # print(data)
        # print(data['train'])
        # print(data['test'])
        if task == "train":
            model(data)
        if task == "visualize":
            l = len(data['train']) + len(data['test'])
            cnt = 1
            # print(np.array(data['train'][0]['input']).shape)
            # print(np.array(data['train'][0]['output']).shape)
            if np.array(data['train'][0]['input']).shape != np.array(data['train'][0]['output']).shape:
                continue
            for i in range(len(data['train'])):
                elem = data['train'][i]
                inp = np.array(elem['input'])
                out = np.array(elem['output'])

                # print("Example ",i+1,", Loss: ",trainer.evaluate(inp,inp,out))
                # print("Example ",i+1,", Loss: ",trainer.evaluate(inp,out,out))
                # pred = np.array(inp)
                # pred[1:3,2:4] = out[1:3,2:4]
                # print("Example ", i + 1, ", Loss: ", trainer.evaluate(inp, pred, out))

                plt.subplot(l,2,cnt)
                # if i == 0:
                #     plt.title(inp.shape)
                cnt += 1
                plt.imshow(inp)
                plt.subplot(l, 2, cnt)
                # if i == 0:
                #     plt.title(out.shape)
                cnt += 1
                plt.imshow(out)

            for i in range(len(data['test'])):
                elem = data['test'][i]
                inp = np.array(elem['input'])
                out = np.array(elem['output'])

                plt.subplot(l, 2, cnt)
                cnt += 1
                if i == 0:
                    plt.title(inp.shape)
                plt.imshow(inp)
                plt.subplot(l, 2, cnt)
                cnt += 1
                if i == 0:
                    plt.title(out.shape)
                plt.imshow(out)
    if task == "visualize":
        plt.show()