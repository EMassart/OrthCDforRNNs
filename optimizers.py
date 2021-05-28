# -*- coding: utf-8 -*-

"""
This file contains the implementation of the optimizers described in the paper "Coordinate descent on the orthogonal group for recurrent
neural network training".
@version: May 2021
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import numpy as np
import time
import random


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class RGD(Optimizer):
    ''' Implementation of stochastic Riemannian gradient descent for orthogonal RNN training.
    Each parameter is supposed to have an "orth" attribute. If param.orth = 1, the parameter is assumed to
    lie on the orthogonal group, and is updated using a Riemannian gradient descent step on the orthogonal group.
    Otherwise, a simple SGD update is used.
    version: May 2021'''

    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(RGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        loss = None
        for p in group['params']:
            d_p = p.grad
            if not p.orth:
                p.data.add_(-group['lr']*d_p.data)
            else:
                A = torch.mm(torch.transpose(p.data,0,1),d_p.data)
                Riema_grad = 0.5*torch.add(A,torch.transpose(A,0,1),alpha = -1)    #Actually the Riemannian gradient is Q*skew(Q'*nabla f(Q)) but we do not need to compute it
                p.data = torch.mm(p.data,torch.matrix_exp(-group['lr']*Riema_grad))
        return loss


class RCD_GS(Optimizer):
    ''' Implementation of stochastic Riemannian coordinate descent for DNN training, using the Gauss-Southwell selection rule
    for selecting the coordinate at each iteration. 
    Each parameter is supposed to have an "orth" attribute. If param.orth = 1, the parameter is assumed to lie on the orthogonal group, and updated 
    using a Riemannian coordinate descent step on the orthogonal group. Otherwise, a simple SGD update is used.
    version: May 2021'''
    
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(RCD_GS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        loss = None
        for p in group['params']:
            d_p = p.grad
            if not p.orth:
                p.data.add_(-group['lr']*d_p.data)
            else:
                A = torch.mm(torch.transpose(p.data,0,1),d_p.data)
                Riema_grad = 0.5*torch.add(A,torch.transpose(A,0,1),alpha = -1)
                Omega = torch.triu(torch.abs(Riema_grad), diagonal=1)
                m,indx = torch.max(Omega,0)
                m2,j2 =  torch.max(m,0)
                j1 = indx[j2]
                alpha = -group['lr']*Riema_grad[j1][j2] 
                v = torch.cos(alpha)*p.data[:,j1] - torch.sin(alpha)*p.data[:,j2]
                w = torch.sin(alpha)*p.data[:,j1] + torch.cos(alpha)*p.data[:,j2]
                p.data[:,j1] = v
                p.data[:,j2] = w
        return loss



class RCD_GS_block(Optimizer):
    ''' Implementation of stochastic Riemannian block coordinate descent for DNN training, using the Gauss-Southwell selection rule
    for selecting the coordinates at each iteration. The parameter nu is the number of coordinates that are selected and updated at each iteration.
    Each parameter is supposed to have an "orth" attribute. If param.orth = 1, the parameter is assumed to lie on the orthogonal group, and updated 
    using a Riemannian block coordinate descent step on the orthogonal group. Otherwise, a simple SGD update is used.
    Note that, since the step parametrization relying on Givens matrices requires coordinates to affect different columns, this implementation still 
    relies on the matrix exponential operation. A smarter coordinate selection strategy should be implemented in order to take benefits from the 
    Givens representation of the update rule.
    version: May 2021'''
    
    def __init__(self, params, lr, nu):
        defaults = dict(lr=lr, nu=nu)
        super(RCD_GS_block, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        loss = None
        for p in group['params']:
            d_p = p.grad
            if not p.orth:
                p.data.add_(-group['lr']*d_p.data)
            else:
                A = torch.mm(torch.transpose(p.data,0,1),d_p.data)
                Riema_grad = 0.5*torch.add(A,torch.transpose(A,0,1),alpha = -1)
                Omega = torch.triu(torch.abs(Riema_grad), diagonal=1)
                Omega2 = torch.zeros_like(Omega)
                Omega = torch.add(Omega,-torch.eye(p.data.size()[0]).to(device))
                for i in range(group['nu']):
                    m,indx = torch.max(Omega,0)
                    m2,j2 =  torch.max(m,0)
                    j1 = indx[j2]
                    Omega2[j1][j2] = Riema_grad[j1][j2]
                    Omega2[j2][j1] = Riema_grad[j2][j1]
                    Omega[j1][j2] = -1
                    Omega[j2][j1] = -1
                p.data = torch.mm(p.data,torch.matrix_exp(-group['lr']*Omega2))
        return loss


class RCD_uniform(Optimizer):
    ''' Implementation of stochastic Riemannian coordinate descent for DNN training, using the uniform selection rule
    for selecting the coordinate at each iteration. 
    Each parameter is supposed to have an "orth" attribute. If param.orth = 1, the parameter is assumed to lie on the orthogonal group, and updated 
    using a Riemannian coordinate descent step on the orthogonal group. Otherwise, a simple SGD update is used.
    version: May 2021'''
    
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(RCD_uniform, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        loss = None
        for p in group['params']:
            d_p = p.grad
            if not p.orth:
                p.data.add_(-group['lr']*d_p.data)
            else:
                A = torch.mm(torch.transpose(p.data,0,1),d_p.data)
                Riema_grad = 0.5*torch.add(A,torch.transpose(A,0,1),alpha = -1)
                b = torch.randperm(p.data.size()[0])
                b0 = b[0]
                b1 = b[1]
                alpha = -group['lr']*Riema_grad[b0][b1] 
                v = torch.cos(alpha)*p.data[:,b0] - torch.sin(alpha)*p.data[:,b1]
                w = torch.sin(alpha)*p.data[:,b0] + torch.cos(alpha)*p.data[:,b1]
                p.data[:,b0] = v
                p.data[:,b1] = w
        return loss

