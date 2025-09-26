import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import CenterCrop

from unet.unet import UNet
from dataset import NeuronalStructures, TrainTransform, TestTransform


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
    ) -> None:
    model.train()
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = F.pad(x, pad=(PADDING, PADDING, PADDING, PADDING), mode=PADDING_MODE)
        y_pred = model(x)
        y_pred = CenterCrop(size=y.shape[-2:])(y_pred)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0 
    for x, y in dataloader:
        N_batch = x.shape[0]
        x = F.pad(x, pad=(PADDING, PADDING, PADDING, PADDING), mode=PADDING_MODE)
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x)
        y_pred = CenterCrop(size=y.shape[-2:])(y_pred)
        loss = criterion(y_pred, y)
        total_loss += loss.item() * N_batch
    return total_loss


def load_train_val(batch_size: int=1, train_size: float=0.8, seed: int=42) -> tuple:
    torch.manual_seed(seed) 
    
    train_transform = TrainTransform()
    dataset_train = NeuronalStructures(transform=train_transform)
    
    test_transform = TestTransform() 
    dataset_test = NeuronalStructures(transform=test_transform)
    
    train_size = int(train_size * len(dataset_train))

    indicies = torch.randperm(len(dataset_train))
    train_idx = indicies[:train_size]
    val_idx = indicies[train_size:]

    train_set = Subset(dataset_train, train_idx)
    val_set = Subset(dataset_test, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True) 

    return train_loader, val_loader


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float('lr', 1e-6, 1e-1)
    momentum = trial.suggest_float('momentum', 0.8, 0.999)

    train_loader, val_loader = load_train_val(BATCH_SIZE, TRAIN_SIZE, SEED)

    model = UNet(in_channels=1, out_channels=2).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum)
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion)
    
    mean_negative_log_likelihood = evaluate(model, val_loader, criterion)
    return mean_negative_log_likelihood


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # --- Settings and Hyperparameters ---
    N_TRIALS = 30
    EPOCHS = 5
    BATCH_SIZE = 1
    TRAIN_SIZE = 0.85
    PADDING = 96
    PADDING_MODE = 'reflect'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    set_seed(SEED)

    # --- Start hyperparameter search --- 
    study = optuna.create_study()
    study.optimize(objective, n_trials=N_TRIALS)