import torch
import torch.nn as nn


import torch.nn as nn

def single_neuron_forward(x):
    # 1. create the layer
    neuron = nn.Linear(3, 1)
    
    # 2. fix the weights inside torch.no_grad()
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([[0.5, -0.2, 0.3]]))
        neuron.bias.copy_(torch.tensor([0.1]))
    
    # 3. run forward and return as float
    return float(neuron(x).item())
