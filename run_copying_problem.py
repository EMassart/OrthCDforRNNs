# -*- coding: utf-8 -*-

"""
This is the final code, with correct seed, to replicate the experiments of our Neurips paper. This code heavily relies (possibly verbatim, mainly regarding 
model architecture and problem setting) on implementations from the project the projects https://github.com/Lezcano/geotorch and https://github.com/Lezcano/expRNN, 
associated to a MIT license. 

@version: May 2021
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import numpy as np
import optimizers
import time
import random

batch_size = 128                                                             
hidden_size = 190                                                            
iterations = 501          #Training iterations                                        
L = 1000                  # Length of sequence before asking to remember
K = 10                    # Length of sequence to remember
n_classes = 9             # Number of possible classes
n_characters = n_classes + 1
n_len = L + 2 * K
lr = 0.0002               # stepsize

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1111)
np.random.seed(1111)
random.seed(1111)


def copy_data(batch_size):
    ''' This code comes is extracted from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment'''
    # Generates some random synthetic data
    # Example of input-output sequence
    # 14221----------:----
    # ---------------14221
    # Numbers go from 1 to 8
    # We generate K of them and we have to recall them
    # L is the waiting between the last number and the
    # signal to start outputting the numbers
    # We codify `-` as a 0 and `:` as a 9.

    seq = torch.randint(1, n_classes, (batch_size, K), dtype=torch.long, device=device)
    zeros1 = torch.zeros((batch_size, L), dtype=torch.long, device=device)
    zeros2 = torch.zeros((batch_size, K - 1), dtype=torch.long, device=device)
    zeros3 = torch.zeros((batch_size, K + L), dtype=torch.long, device=device)
    marker = torch.full((batch_size, 1), n_classes, dtype=torch.long, device=device)
    x = torch.cat([seq, zeros1, marker, zeros2], dim=1)
    y = torch.cat([zeros3, seq], dim=1)
    return x, y



class modrelu(nn.Module):
    ''' This code comes is extracted from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment'''
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)
        return phase * magnitude


def henaff_init_(A):
    ''' This code comes is extracted from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment'''
    # I needed to update Henaff initialization as we don't want the skew-symmetric variable but the orthogonal one....
    size = A.size(0) // 2
    diag = A.new(size).uniform_(-np.pi, np.pi)
    A_init = create_diag_(A, diag)
    I = torch.eye(A_init.size(0))
    return torch.mm(torch.inverse(I+A_init), I-A_init)
    #return torch.matrix_exp(A_init)


def create_diag_(A, diag):
    ''' This code comes is extracted from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment'''
    n = A.size(0)
    diag_z = torch.zeros(n-1)
    diag_z[::2] = diag
    A_init = torch.diag(diag_z, diagonal=1)
    A_init = A_init - A_init.T
    with torch.no_grad():
        A.copy_(A_init)
        return A



class FlexibleRNN(nn.Module):
    ''' This code comes is extracted (and slightly modified) from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment'''

    def __init__(self, input_size, hidden_size):
        super(FlexibleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.input_kernel = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=False)
        self.nonlinearity = modrelu(hidden_size)
        #self.nonlinearity = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")
        nn.init.zeros_(self.recurrent_kernel.weight.data)
        self.recurrent_kernel.weight.data = henaff_init_(self.recurrent_kernel.weight.data)
        #self.recurrent_kernel.weight.data = nn.init.orthogonal_(self.recurrent_kernel.weight)

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        out = self.nonlinearity(out)
        return out, out



class Model(nn.Module):
    ''' This code comes is extracted from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment'''
    def __init__(self, n_classes, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        #self.rnn = nn.LSTMCell(n_classes + 1, hidden_size)
        self.rnn = FlexibleRNN(n_classes + 1, hidden_size)
        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.CrossEntropyLoss()
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.lin.weight.data, nonlinearity="relu")
        nn.init.constant_(self.lin.bias.data, 0)

    def forward(self, inputs):     
        state = torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device)
        outputs = []
        for input in torch.unbind(inputs, dim=1):
            out_rnn, state = self.rnn(input, state)
            if isinstance(self.rnn, nn.LSTMCell):
                state = (out_rnn, state)
            outputs.append(self.lin(out_rnn))
        return torch.stack(outputs, dim=1)  

    def loss(self, logits, y):
        return self.loss_func(logits.view(-1, 9), y.view(-1))

    def accuracy(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=2), y).float().mean()



def onehot(out, input):
    ''' This code comes is extracted from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment'''
    out.zero_()
    in_unsq = torch.unsqueeze(input, 2)
    out.scatter_(2, in_unsq, 1)

model = Model(n_classes, hidden_size).to(device)
p_orth = model.rnn.recurrent_kernel
for p in model.parameters():
    if p in set(p_orth.parameters()):
        p.orth = 1
    else:
        p.orth = 0

x_onehot = torch.FloatTensor(batch_size, n_len, n_characters).to(device)
optim = optimizers.RCD_uniform(model.parameters(), lr=lr)

train_loss_record = []
train_time_record = []
train_accuracy_record = []
t0_rgd = time.time()

for step in range(iterations):

    batch_x, batch_y = copy_data(batch_size)
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    onehot(x_onehot, batch_x)
    logits  = model(x_onehot)

    loss = model.loss(logits.view(-1, 9), batch_y.view(-1))
    train_loss_record.append(loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()

    # # save stochastic gradient at initialization
    # if step==0:
    #     orth_par_init = model.rnn.recurrent_kernel.weight.data.clone().detach()
    #     orth_par_grad_init = model.rnn.recurrent_kernel.weight.grad.data.clone().detach()
    #     with open('grad_init.npy', 'wb') as f:
    #         np.save(f, orth_par_init.cpu())
    #         np.save(f, orth_par_grad_init.cpu())

    # if step==500:
    #     orth_par_end = model.rnn.recurrent_kernel.weight.data.clone().detach()
    #     orth_par_grad_end = model.rnn.recurrent_kernel.weight.grad.data.clone().detach()
    #     with open('grad_end.npy', 'wb') as f:
    #         np.save(f, orth_par_end.cpu())
    #         np.save(f, orth_par_grad_end.cpu())


    with torch.no_grad():
        accuracy = model.accuracy(logits, batch_y)
        
    elapsed_time = time.time()-t0_rgd
    train_accuracy_record.append(accuracy.item())
    train_time_record.append(elapsed_time)


    if step%10==0:
        print('Iteration', step, 'loss', loss.item(), 'accuracy', accuracy.item())


with open('results_copying_problem.npy', 'wb') as f:
    np.save(f, train_loss_record)
    np.save(f, train_accuracy_record)
    np.save(f, train_time_record)