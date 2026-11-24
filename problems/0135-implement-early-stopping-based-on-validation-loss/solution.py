from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Your code here
    best_loss = val_losses[0]
    best_epoch = 0
    counter = 0

    for epoch in range(1, len(val_losses)):
        current_loss = val_losses[epoch]
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_epoch = epoch
            counter = 0
        else :
            counter += 1 
            if counter >= patience:
                return epoch, best_epoch
    stop_epoch = len(val_losses) - 1
    return (stop_epoch, best_epoch)
