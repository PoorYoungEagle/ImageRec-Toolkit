import torch
import tqdm
from typing import Tuple

def train_epoch_ce(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> Tuple[float, float]:
    """
    Runs one training epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
        device (torch.device): Device to run computations on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: (average_loss, accuracy)
            - average_loss (float): Mean training loss across the epoch.
            - accuracy (float): Training accuracy (%) across the epoch.
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm.tqdm(train_loader, desc = "Train", leave = False)

    for batch_index, (inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix({'Loss': f'{running_loss/(batch_index+1):.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

    average_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
    accuracy = 100. * correct / total if total > 0 else 0.0

    return average_loss, accuracy


def validate_epoch_ce(
        model: torch.nn.Module,
        validation_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device
) -> Tuple[float, float]:
    """
    Runs one validation epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        validation_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run computations on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: (average_loss, accuracy)
            - average_loss (float): Mean validation loss across the epoch.
            - accuracy (float): Validation accuracy (%) across the epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm.tqdm(validation_loader, desc = "Validation", leave = False)
        for batch_index, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix({'Loss': f'{running_loss/(batch_index+1):.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    
    average_loss = running_loss / len(validation_loader) if len(validation_loader) > 0 else float('nan')
    accuracy = 100. * correct / total if total > 0 else 0.0

    return average_loss, accuracy


def train_epoch_bce(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> Tuple[float, float]:
    """
    Runs one training epoch for binary classification with BCE loss.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): BCE loss function (e.g., BCELoss or BCEWithLogitsLoss).
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
        device (torch.device): Device to run computations on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: (average_loss, accuracy)
            - average_loss (float): Mean training loss across the epoch.
            - accuracy (float): Training accuracy (%) across the epoch.
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm.tqdm(train_loader, desc="Train", leave=False)

    for batch_index, (inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if outputs.dim() > 1 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)
        
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        
        total += targets.size(0)
        correct += (predicted == targets.float()).sum().item()

        progress_bar.set_postfix({'Loss': f'{running_loss/(batch_index+1):.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

    average_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
    accuracy = 100. * correct / total if total > 0 else 0.0

    return average_loss, accuracy


def validate_epoch_bce(
        model: torch.nn.Module,
        validation_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device
) -> Tuple[float, float]:
    """
    Runs one validation epoch for binary classification with BCE loss.

    Args:
        model (torch.nn.Module): The model to evaluate.
        validation_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): BCE loss function (e.g., BCELoss or BCEWithLogitsLoss).
        device (torch.device): Device to run computations on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: (average_loss, accuracy)
            - average_loss (float): Mean validation loss across the epoch.
            - accuracy (float): Validation accuracy (%) across the epoch.
    """
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm.tqdm(validation_loader, desc="Validation", leave=False)
        
        for batch_index, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, targets.float())
            running_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            total += targets.size(0)
            correct += (predicted == targets.float()).sum().item()

            progress_bar.set_postfix({'Loss': f'{running_loss/(batch_index+1):.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    
    average_loss = running_loss / len(validation_loader) if len(validation_loader) > 0 else float('nan')
    accuracy = 100. * correct / total if total > 0 else 0.0

    return average_loss, accuracy