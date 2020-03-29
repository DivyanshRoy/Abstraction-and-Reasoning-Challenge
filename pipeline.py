import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from os import listdir
from os.path import isfile, join
matplotlib.use('TkAgg')
import trainer

task = "train"
# task = "visualize"

path = "ARC-master/data/training/"

filenames = [f for f in listdir(path) if isfile(join(path, f))]
correct = 0
for filename in filenames:
    with open(path+"/"+filename) as json_file:
        data = json.load(json_file)
        if task == "train":
            train_data = {"input": [], "output": []}
            eval_data = {"input": [], "output": []}

            for i in range(len(data['train'])):
                elem = data['train'][i]
                inp = np.array(elem['input'])
                out = np.array(elem['output'])
                train_data["input"].append(inp)
                train_data["output"].append(out)

            for i in range(len(data['test'])):
                elem = data['test'][i]
                inp = np.array(elem['input'])
                out = np.array(elem['output'])
                eval_data["input"].append(inp)
                eval_data["output"].append(out)

            trainer.train(train_data["input"], train_data["output"])
            prediction = trainer.predict(eval_data["input"])
            loss = trainer.evaluate(input=eval_data["input"], prediction=prediction, ground_truth=eval_data["output"])

        if task == "visualize":
            l = len(data['train']) + len(data['test'])
            cnt = 1
            if np.array(data['train'][0]['input']).shape != np.array(data['train'][0]['output']).shape:
                continue
            for i in range(len(data['train'])):
                elem = data['train'][i]
                inp = np.array(elem['input'])
                out = np.array(elem['output'])
                plt.subplot(l,2,cnt)
                cnt += 1
                plt.imshow(inp)
                plt.subplot(l, 2, cnt)
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