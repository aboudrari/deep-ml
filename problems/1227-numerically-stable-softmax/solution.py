import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    max_val = t.max(dim = dim, keepdim = True).values
    shifted = t - max_val
    exp = torch.exp(shifted)
    sum_exp = exp.sum(dim = dim, keepdim= True)
    return exp / sum_exp
