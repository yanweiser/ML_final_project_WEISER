import numpy as np
import json
from datetime import datetime
import os

import utils


class Perceptron:
    def __init__(self, config):
        self.weights = np.zeros(config['weigth_size'])

    def forward(self, x):
        out = self.weights.dot(x)
        return out

    def backward(self, pred, label, x):
        loss = 0  
        if pred*label <= 0:
            self.weights += label*x
            loss = -pred*label
        return loss
    

def train_perceptron(model, train_data, test_data, max_epochs):
    losses = []
    train_accs = []
    test_accs = []
    length = len(train_data[1])
    for i in range(max_epochs):
        epoch_loss = 0
        correct = 0
        last = np.copy(model.weights)
        for j in range(length):
            pred = model.forward(train_data[0][j])
            loss = model.backward(pred, train_data[1][j], train_data[0][j])
            if pred * train_data[1][j] > 0:
                correct += 1
            epoch_loss += loss
        losses.append(epoch_loss)
        test_count = 0
        for l in range(len(test_data[1])):
            pred = model.forward(test_data[0][l])
            if pred * test_data[1][l] > 0:
                test_count += 1
        test_accs.append(test_count/len(test_data[1]))
        train_accs.append(correct/length)
        if i%5 == 0:
            print(i, ':  epoch loss = ', epoch_loss, '\t Accuracy = ', correct/length)
        if all(np.equal(model.weights, last)):
            print('stopping in epoch {} due to no changes'.format(i))
            break
    log_data = {'losses': losses, 'train_accs': train_accs, 'test_accs': test_accs}   
    now = datetime.now()  
    date_time = now.strftime("%Y_%m_%d_%H_%M")
    with open(os.path.join('results', f'perceptron_training_{date_time}.json'), 'w') as f:
        json.dump(log_data, f)

    ##### testing #####
    test_perceptron(model, train_data, test_data)


def test_perceptron(model, train_data, test_data):
    print('\nTest Data:')
    length = len(test_data[1])
    preds = []
    for i in range(length):
        pred = model.forward(test_data[0][i])
        preds.append(pred)
    utils.analyze(preds, test_data[1])
    print('\nAll the Data:')
    all_x = np.concatenate((train_data[0], test_data[0]))
    all_y = np.concatenate((train_data[1], test_data[1]))
    length = len(all_y)
    preds = []
    for i in range(length):
        pred = model.forward(all_x[i])
        preds.append(pred)
    utils.analyze(preds, all_y)
        