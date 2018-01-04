# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 17:02:11 2017

@author: Cybot & Pibotinus
"""

import numpy as np


# Activation function
def sigmoid(inp):
    """Sigmoid unit"""
    return 1/(1+np.exp(-inp))

# For Back Propagation
def derv(inp):
    """Back propagation differential"""
    return (inp*(1-inp))

np.random.seed(1)

# Synapses for two layers
SYNAPSES0 = 2*np.random.random((3, 4)) - 1
SYNAPSES1 = 2*np.random.random((4, 1)) - 1

def forward(inp):
    """Forwarding function"""
    layer1 = sigmoid(np.dot(inp, SYNAPSES0))
    layer2 = sigmoid(np.dot(layer1, SYNAPSES1))
    return layer1, layer2

def error(layer1, layer2, label):
    """Error function"""
    layer2_err = label - layer2
    layer2_delta = layer2_err*derv(layer2)
    layer1_err = layer2_delta.dot(SYNAPSES1.T)
    layer1_delta = layer1_err * derv(layer1)
    return layer1_delta, layer2_delta


def update(layer1, layer1_delta, layer2_delta, inp):
    """Update function"""
    global SYNAPSES0
    global SYNAPSES1
    SYNAPSES1 += layer1.T.dot(layer2_delta)
    SYNAPSES0 += inp.T.dot(layer1_delta)

# Training phase
def training():
    """Function which allows the model to train"""
    # Input as exclusive XOR
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # Labels
    Y = np.array([[0], [1], [1], [0]])
    for j in range(100000):
        # Feed Forward
        layer1, layer2 = forward(X)
        # Errors and deltas for BP  
        layer1_delta, layer2_delta = error(layer1, layer2 ,Y)
        # Synapses update
        update(layer1, layer1_delta, layer2_delta, X)
    # Results

    print "[+] Training completed!"
    print layer2

def testing():
    """Testing function"""
    print "[!] To calculate a XOR of 3 bit"
    while 1:
        data = raw_input("Please provide 3 spaced bits or enter q to quit --> ")
        if(data == 'q'):
            break
        inp = np.array([map(int, data.split(' '))])
        layer1, output = forward(inp)
        print str(output)

def main():
    """Main function"""
    print "[!] Hello dear user !"
    training()
    testing()

if __name__ == "__main__":
    main()
