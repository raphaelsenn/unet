import random
import argparse
from argparse import Namespace

import optuna

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

from unet.unet import UNet
from unet.objective import CrossEntropyLossWithWeights
from unet.dataset import NeuronalStructuresAugmented


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
    ) -> None:
    model.train()
    for x, y, w in dataloader:
        x, y, w = x.to(DEVICE), y.to(DEVICE).long(), w.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y, w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    N_data = 0
    for x, y in dataloader:
        N_batch = x.shape[0]
        x, y = x.to(DEVICE), y.to(DEVICE).long()
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * N_batch
        N_data += 1 
    total_loss = total_loss / N_data
    return total_loss


def objective(trial: optuna.Trial) -> float:
    # --- Suggest hyperparameters ---
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    momentum = 0.99
    gamma = trial.suggest_float('gamma', 0.97, 0.999)

    # --- Load training and validation data ---
    train_set = NeuronalStructuresAugmented(train=True)
    indices_train = torch.randperm(len(train_set))[:N_TRAIN_SAMPLES]
    train_set = Subset(train_set, indices_train)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    val_set = NeuronalStructuresAugmented(subdir_features='val/images', subdir_labels='val/masks', landmarks='landmarks_val.csv', train=False)
    indices_val = torch.randperm(len(val_set))[:N_VAL_SAMPLES]
    val_set = Subset(val_set, indices_val) 
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # --- Initialize model, criterion, losss, ... ---
    model = UNet(in_channels=1, out_channels=2).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum)
    criterion = CrossEntropyLossWithWeights().to(DEVICE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    # --- Start training ---
    for _ in range(EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

    return evaluate(model, val_loader, criterion)


def set_seed(seed: int = 42) -> None:
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog='UNet Hyperparameter Search: ISBI-2012 Neuronal Structure Segmentation',
        description='Search hyperparameters of the UNet model on the ISBI-2012 dataset for neuronal structure segmentation.'
    )
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--epochs_per_trial', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    # --- Parse arguments ---
    args = parse_args() 

    # --- Settings and Fixed-Hyperparameters ---
    N_TRIALS = args.n_trials
    EPOCHS = args.epochs_per_trial
    N_TRAIN_SAMPLES = 500
    N_VAL_SAMPLES = 50
    BATCH_SIZE = 1
    SEED = args.seed
    DEVICE = torch.device(args.device)
    set_seed(SEED)

    # --- Start hyperparameter search --- 
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)