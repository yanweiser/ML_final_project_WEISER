import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def analyze(preds, labels):
    '''
    calculates false and true positives and negatives as well as their rates.
    Also calculates precision, Recall and Accuracy and prints these values in a 
    well formatted string.
    '''
    tp, fp, tn, fn = 0, 0, 0, 0
    length = len(preds)
    for i in range(length):
        p = preds[i]
        l = labels[i]
        if p > 0 and l > 0:
            tp += 1
        elif p > 0 and l < 0:
            fp += 1
        elif p < 0 and l < 0:
            tn += 1
        elif p < 0 and l > 0:
            fn += 1 
    accuracy = (tp + tn)/length
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fpr = fp/(fp+tn)
    output = ""
    output +="=== Prediction Analysis ===\n"

    output +=f"data points:       {length}\n"
    output +="-----------------------------\n"
    output +=f"true positives:    {tp} ({tp/length:.5f})\n"
    output +=f"false positives:   {fp} ({fp/length:.5f})\n"
    output +=f"true negatives:    {tn} ({tn/length:.5f})\n"
    output +=f"false negatives:   {fn} ({fn/length:.5f})\n"
    output +="-----------------------------\n"
    output +=f"accuracy:          {accuracy:.5f}\n"
    output +=f"precision:         {precision:.5f}\n"
    output +=f"recall (tpr):      {recall:.5f}\n"
    output +=f"fpr:               {fpr:.5f}"
    print(output)
    return

def accuracy(preds, labels):
    '''
    the threshold for classification is 0.5 (since binary Cross Entropy only takes values between 0 and 1)
    therefore the results and labels are shifted down by 0.5 so they can be classified by their sign.
    
    Inputs:
        preds: torch tensor (float32) [0,1]
        labels: torch tensor (float32) [0,1]
    Output:
        rate of correct prediction as float value
    '''
    equals = torch.eq(torch.sign(preds-0.5), torch.sign(labels-0.5))
    summed = torch.sum(equals, dim=0)
    acc = summed.item()/len(labels)
    return acc

def run_analysis(model, train_loader, test_loader, device):
    with torch.no_grad():
        outs = []
        outs_train = []
        labels = []
        labels_train = []
        for x,y in test_loader:
            part = model(x.to(device))
            part = part.view(-1)
            outs.append(part.cpu().numpy())
            labels.append(y.numpy())
        for x,y in train_loader:
            part = model(x.to(device))
            part = part.view(-1)
            outs_train.append(part.cpu().numpy())
            labels_train.append(y.numpy())
        preds = np.hstack(tuple(outs))
        labels = np.hstack(tuple(labels))
        preds = np.sign(preds-0.5)
        labels = np.sign(labels-0.5)
        preds_train = np.hstack(tuple(outs_train))
        labels_train = np.hstack(tuple(labels_train))
        preds_train = np.sign(preds_train-0.5)
        labels_train = np.sign(labels_train-0.5)
        preds_all = np.hstack((preds_train, preds))
        labels_all = np.hstack((labels_train, labels))
        print('-'*5, 'Only Test Set', '-'*5)
        analyze(preds, labels)
        print('-'*5, 'All the Data', '-'*5)
        analyze(preds_all, labels_all)


class Dataset(Dataset):
    '''
    very simple Dataset that is only nessecary as input for the dataLoader.  
    '''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = len(Y)

    def __getitem__(self, idx):
        x_ret = self.X[idx]
        y_ret = self.Y[idx]

        return x_ret, y_ret

    def __len__(self):
        return self.len
