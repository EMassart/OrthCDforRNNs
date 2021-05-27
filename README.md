# OrthCDforRNNs

This repository contains implementations of stochastic Riemannian coordinate descent (SRCD) on the orthogonal group for training recurrent neural networks, described in the paper "Coordinate descent on the orthogonal group for recurrent neural network training", by V. Abrol and E. Massart, 2021.

In the proposed algorithm, the recurrent matrix of the network is subject to orthogonal constraints, while the other network parameters are free. In the proposed optimizers, the orthogonal parameter is updated using stochastic Riemannian coordinate descent on the orhtogonal group, while the free parameters are simply updated using stochastic gradient descent. Coordinate descent on the orthogonal group amounts to apply successive rotations to two columns of the recurrent matrix, and operation that can easily be implemented in terms of right-multiplications by Givens matrices.

We propose here several variants of our SRCD algorithm presented in the paper, all implemented in the file optimizer.py:
- an implementation of SRCD in which the coordinates are selected uniformly at random
- an implementation of SRCD endowed with the Gauss-Southwell coordinate selection rule
- a block coordinate counterpart of SRCD with Gauss-Soutwell
- as a comparison point, an implementation of stochastic Riemannian gradient descent. 

The file run_copying_problem.py allows to replicate the results given in the paper for the copying task. 
