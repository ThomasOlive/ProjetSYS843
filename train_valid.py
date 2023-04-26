import torch
import optuna


def train_valid(model, optimizer, criterion, n_epochs, trainloader, validloader, device, trial):
    # _________ Training ___________________________
    # Lists to store training and validation losses
    t_losses, v_losses = [], []
    # Loop over epochs
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0.0, 0.0

        # train step
        model.train()
        # Loop over train dataset
        for x, y in trainloader:
            optimizer.zero_grad()
            # move inputs to device
            x = x.to(device)
            y = y.squeeze().to(device)

            # Forward Pass
            preds = model(x).squeeze()
            loss = criterion(preds, y)  # compute batch loss
            # loss = criterion(preds, y) + regul(model)
            train_loss += loss.item()  # added to a list
            loss.backward()
            optimizer.step()
        epoch_loss = train_loss / len(trainloader)  # mean of the list is the epoch loss
        t_losses.append(epoch_loss)

        # validation step
        model.eval()
        # Loop over validation dataset
        for x, y in validloader:
            with torch.no_grad():
                x = x.to(device)
                y = y.squeeze().to(device)
                preds = model(x).squeeze()
                error = criterion(preds, y)  # computed for each batch
                # error = criterion(preds, y) + regul(model)
            valid_loss += error.item()  # added to a list
        valid_loss = valid_loss / len(validloader)  # mean of the list is the epoch loss
        v_losses.append(valid_loss)

        print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')

        # trial.report(valid_loss, epoch)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    return model, t_losses, v_losses

