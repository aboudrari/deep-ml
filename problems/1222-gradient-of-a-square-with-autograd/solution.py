import torch

def grad_of_square(x_val):
    x = torch.tensor(x_val, requires_grad=True)
    y = x ** 2
    y.backward()
    return x.grad.item()