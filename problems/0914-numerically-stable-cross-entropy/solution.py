import torch

def cross_entropy(logits, targets):
    log_z = torch.logsumexp(logits, dim=1)

    correct_logits = logits[
        torch.arange(logits.shape[0]),
        targets
    ]

    loss = -correct_logits + log_z

    return loss.mean()
