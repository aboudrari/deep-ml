def early_stopping(val_losses, patience, min_delta):
    best_loss = val_losses[0]
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, len(val_losses)):
        current_loss = val_losses[epoch]

        if best_loss - current_loss > min_delta:
            best_loss = current_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement == patience:
            return (epoch, best_epoch)

    return (len(val_losses) - 1, best_epoch)

