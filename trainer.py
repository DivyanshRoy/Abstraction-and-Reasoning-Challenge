import numpy as np

def train(inputs, outputs):
    #TODO
    pass

def evaluate(input, prediction, ground_truth):
    if prediction.shape != ground_truth.shape:
        print("Improper shape")
        loss = 1
        return loss
    diff = np.abs(prediction-ground_truth)
    if np.sum(diff) == 0:
        loss = 0
        return loss
    else:
        diff2 = np.abs(input-ground_truth)
        diff2 = np.minimum(diff2, 1)
        diff = np.minimum(diff, 1)
        loss = np.sum(diff)/np.sum(diff2)
        # loss = np.sum(diff)/(diff.shape[0]*diff.shape[1])
        return loss


