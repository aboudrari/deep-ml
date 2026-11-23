import torch
import torch.nn as nn
import torch.nn.functional as F

def train_step(model, x_batch, y_batch, lr):

    # 1. Zero gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    # 2. Forward
    logits = model(x_batch)

    # 3. Loss
    loss = F.cross_entropy(logits, y_batch)

    # 4. Backpropagation
    loss.backward()

    # 5. Update weights
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

    # 6. Return Python float
    return loss.item()
