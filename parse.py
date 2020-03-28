import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

with open('example.json') as json_file:
    data = json.load(json_file)
    # print(data)
    print(data['train'])
    print(data['test'])
    l = len(data['train']) + len(data['test'])
    cnt = 1
    print(np.array(data['train'][0]['input']).shape)
    print(np.array(data['train'][0]['output']).shape)
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
        plt.imshow(inp)
        plt.subplot(l, 2, cnt)
        cnt += 1
        plt.imshow(out)

plt.show()