# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 17:02:11 2017

@author: Cybot
"""

import numpy as np


# Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# For Back Propagation
def derv(x):
    return (x*(1-x))


# Input as exclusive XOR
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

# Labels
y = np.array([[0],[1],[1],[0]])

np.random.seed(1)

# Synapses for two layers
synapse0 = 2*np.random.random((3,4)) - 1
synapse1 = 2*np.random.random((4,1)) - 1 


# Training phase
for j in range(100000):

    # Feed Forward

    layer0 = X
    layer1 = sigmoid(np.dot(layer0,synapse0))
    layer2 = sigmoid(np.dot(layer1,synapse1))

    # Errors and deltas for BP

    layer2_err = y - layer2

    layer2_delta = layer2_err*derv(layer2)

    layer1_err = layer2_delta.dot(synapse1.T)

    layer1_delta = layer1_err * derv(layer1)

    # Synapses update
    synapse1 += layer1.T.dot(layer2_delta)
    synapse0 += layer0.T.dot(layer1_delta)

# Resluts

print("[+] Training completed!")
print(layer2)
