import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    history = []

    # setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n_samples = X_train.shape[0]

    for epoch in range(1, epochs + 1):          # 1-indexed, as the spec wants

        # ---- train ----
        model.train()

        # shuffle (X and y with the SAME perm — flashcards stay married)
        perm = torch.randperm(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        # batch loop
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            # the training step — yours, unchanged
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()           # .item() → plain float, no memory leak
            n_batches += 1

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            preds = val_logits.argmax(dim=1)
            val_accuracy = (preds == y_val).float().mean().item()

        # ---- record ----
        history.append({
            'epoch': epoch,
            'train_loss': epoch_loss / n_batches,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        })

    return history