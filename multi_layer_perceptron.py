import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import os
import numpy as np
import scipy.io
import scipy.sparse
import yaml
import time
import json
import random

import utils

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MLP(nn.Module):
    '''
    This model is a multi-layer Perceptron (MLP) that is used for a classification task.
    The model consists of 4 weight layers:
        (1) The encoder transforms the bag-of-words representations into embeddings, ReLU is used as activation
        (2) and (3) produce hidden states and introduce additional non-linearity, again ReLU is used as activation
        (4) the decoder uses the hidden state to produce a singular float value that is being passed into a logistic function
            to produce a value between 0 and 1. 

    Input Size is N * V where N is the batch size and V the size of the vocabulary.

    The  Output is of size N * 1. 
    '''
    def __init__(self, config):
        super(MLP, self).__init__()

        self.config = config

        self.encoder = nn.Linear(self.config['in_size'], self.config['embed_size'])
        self.hidden = nn.Linear(self.config['embed_size'], self.config['dim_hidden'])
        self.hidden2 = nn.Linear(self.config['dim_hidden'], self.config['reduced_dim_hidden'])
        self.decoder = nn.Linear(self.config['reduced_dim_hidden'], self.config['out_size'])

        self.relu = nn.ReLU()
        self.sigma = nn.Sigmoid()


    def forward(self, inputs):
        
        embeds = self.relu(self.encoder(inputs))
        hidden_state = self.relu(self.hidden(embeds))
        hidden_state2 = self.relu(self.hidden2(hidden_state))
        logits = self.decoder(hidden_state2)
        output = self.sigma(logits)

        return output


def train(model, optimizer, loss_func, dataloader, config, test_data):
    '''
    trains the given model and returns the training performance per epoch
    
    Inputs:
        model: The MLP model to be trained
        optimizer: optimizer used to perform backpropagation
        loss_func: the loss function used (in this case binary Cross Entropy Loss)
        dataloader: Object of type DataLoader used to get batched training data
        config: stores the number of epochs the model should be trained for
        test_data: DataLoader Object that is used to get test data 

    Outputs:
        losses: average loss per epoch (list of floats)
        train_accs: accuracy on the training data each epoch (list of floats)
        test_accs: accuracy on the testing data each epoch (list of floats)
    '''
    train_accs = []
    test_accs = []
    losses = []
    
    for epoch in range(config['epochs']):

        epoch_loss = 0

        for num_batch, (X_batch, y_batch) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            preds = preds.view(-1)# preds is returned as Nx1, this reshapes it into one dimension of size N
            loss = loss_func(preds, y_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(f'({num_batch}/{len(dataloader)}) Loss = {loss.item()}', end='\r')
        avg_loss = epoch_loss/len(dataloader)
        print(f'Epoch {epoch}:    Loss = {avg_loss}')
        with torch.no_grad():
            train_acc = test(model, dataloader, config)
            test_acc = test(model, test_data, config)
            losses.append(avg_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            if epoch%10 == 0:
                print('\tTrain Accuracy = ', train_acc)
                print('\tTest Accuracy = ', test_acc)
    return losses, train_accs, test_accs
            

def test(model, dataloader, config):
    '''
    takes a model and a dataloader and returns the accuracy of the model on the given data
    '''
    with torch.no_grad():
        accs = []

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)
            acc = utils.accuracy(preds.view(-1), y_batch)
            accs.append(acc)
        mean_acc = np.mean(accs)
        # print(f'Accuracy = {mean_acc}')
        return mean_acc

        


if __name__ == '__main__':
    print("Loading data...")
    start = time.time()
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f) # loads the experiment configuration
    torch.manual_seed(config['manual_seed']) # set a manual seed for reproducability
    torch.cuda.manual_seed(config['manual_seed']) # also need to set it for runs on GPU
    random.seed(config['manual_seed']) # additional randomness
    # loads the data using loadmat function
    data = scipy.io.loadmat('emails.mat') 

    # since this data is in a compressed sparse representation
    # it is decompressed using the todense() method. 
    # also the data is in form V * N and it should be N * V
    # so the dimensions are being swapped
    X = np.asarray(data['X'].todense().transpose(1,0), dtype='float32') 
    Y = np.asarray(data['Y'].reshape(-1), dtype='float32') # the labels are not compressed and can be read directly
    Y = (Y+1)/2 # labels are given as [-1, 1] but need to be [0,1] for binary Cross Entropy Loss
    config['in_size'] = X.shape[1]
     # split data into train and test sets, also shuffles the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = config['train_size'], shuffle=True, random_state=config['manual_seed'])
    trainset = utils.Dataset(X_train, Y_train)
    train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    testset = utils.Dataset(X_test, Y_test)
    test_loader = DataLoader(testset, batch_size=config['batch_size'], shuffle=True)
    print("took {:.2f} seconds".format(time.time() - start))
    
    model = MLP(config) # initiallizes MLP model
    model = model.to(device) # only effects GPU environments
    model.train()
    optimizer = optim.Adam(params = model.parameters(), lr=config['lr']) # Adam seems to be a good multi-purpose Optimizer
    loss_fn = nn.BCELoss() # binary Cross Entropy Loss
    losses, train_accs, test_accs = train(model, optimizer, loss_fn, train_loader, config, test_loader) # trains the model and returns results
    if not os.path.isdir('results'):
        os.mkdir('./results')
    log_data = {'losses': losses, 'train_accs': train_accs, 'test_accs': test_accs}
    with open(os.path.join('results', f'{config["manual_seed"]}.json'), 'w') as f:
        json.dump(log_data, f) # log training results
    print()
    model.eval()
    utils.run_analysis(model, train_loader, test_loader, device)
    torch.save(model.state_dict(), os.path.join('models', 'mlp.pth'))
