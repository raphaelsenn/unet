import os
import random
import argparse
from argparse import Namespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision.transforms.functional as TF

from unet.unet import UNet
from unet.objective import CrossEntropyLossWithWeights
from unet.dataset import NeuronalStructuresOriginal, NeuronalStructuresAugmented, TestTransform


def train(
        model: nn.Module,
        dataloader: DataLoader,
        dataloader_val: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: LRScheduler
    ) -> None:
    losses_train = [] 
    losses_val = [] 
    N_saved_images = 0
    for epoch in range(EPOCHS): 
        total_loss = 0.0 
        model.train() 
        counter = 0 
        for x, y, w in dataloader:
            N_batch = x.shape[0]
            x, y, w = x.to(DEVICE), y.to(DEVICE).long(), w.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y, w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * N_batch 

            # --- NOTE: Plotting increases training time ---
            if counter % 100 == 0:
                save_images(model, ORIG_TRAIN_SET, N_saved_images)
                N_saved_images += 1
            counter += 1 
        scheduler.step()

        # --- NOTE: Evaluation increases training time ---
        val_loss = evaluate(model, dataloader_val, criterion)

        losses_train.append(total_loss / len(dataloader.dataset))
        # losses_train_val.append(train_val_loss)
        losses_val.append(val_loss)

        if VERBOSE: 
            total_loss /= len(dataloader.dataset)
            print(f'epoch: {epoch+1}/{EPOCHS}\ttrain_loss: {total_loss:.4f}\tval_loss: {val_loss:.4f}')

    # --- Save losses --- 
    df = pd.DataFrame({'epochs': np.arange(len(losses_train)), 'train_loss': losses_train, 'val_loss': losses_val})
    df.to_csv('losses.csv', index=False)


@torch.no_grad()
def evaluate(model, dataloader, criterion) -> float:
    model.eval() 
    total_loss = 0.0 
    for x, y in dataloader:
        N_batch = x.shape[0]
        x, y = x.to(DEVICE), y.to(DEVICE).long()
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * N_batch
    return total_loss / len(dataloader.dataset)


def set_seed(seed: int = 42) -> None:
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_images(
        model: nn.Module, 
        dataset: Dataset, 
        epoch: int
    ) -> None:
    DPI = 150
    PANEL_PX = 384
    N_COLS = 6
    FIGSIZE = (N_COLS * PANEL_PX / DPI, 3 * PANEL_PX / DPI)
    PLOT_DIR = './imgs_pred'
    
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    
    fig, ax = plt.subplots(
        nrows=3, ncols=N_COLS, figsize=FIGSIZE, dpi=DPI, constrained_layout=True
    )
    titles = ["Input", "Target", "Prediction"]
    for r in range(3):
        ax[r, 0].set_ylabel(titles[r])

    model.eval()
    with torch.no_grad():
        
        for c in range(min(N_COLS, len(dataset))):
            x, y = dataset[c]   # [1, 1, 572, 572], [1, 388, 388]
            if x.ndim == 2: 
                x = x.unsqueeze(0)
            H, W = y.shape[-2], y.shape[-1]
            x_in = x.unsqueeze(0).to(DEVICE)        # [1,1,H,W]

            logits = model(x_in)
            y_hat = logits.softmax(1).argmax(1).squeeze(0).cpu().numpy()

            x_in = TF.center_crop(x_in, output_size=(H, W))     # [1, 388, 388]
            img_np = x_in.cpu().squeeze().numpy()
            y_np   = y.cpu().numpy() if torch.is_tensor(y) else y

            ax[0, c].imshow(img_np, cmap="gray", interpolation="nearest")
            ax[1, c].imshow(y_np,   cmap="gray", interpolation="nearest")
            ax[2, c].imshow(y_hat,  cmap="gray", interpolation="nearest")
            for r in range(3):
                ax[r, c].axis("off")

    out = os.path.join(PLOT_DIR, f"pred_masks_{epoch}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog='UNet Training: ISBI-2012 Neuronal Structure Segmentation',
        description='Train a UNet model on the ISBI-2012 dataset for neuronal structure segmentation.'
    )
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0011261985946982916)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.97578034128803)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    # --- Parse arguments ---
    args = parse_args() 

    # --- Hyperparameters ---
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    MOMENTUM = args.momentum
    GAMMA = args.gamma
    BATCH_SIZE = 1

    # --- Settings ---
    SEED = args.seed
    DEVICE = torch.device(args.device)
    VERBOSE = args.verbose
    SAVE_MODEL_PATH = './unet31m.pth'
    set_seed(SEED)

    # --- Load training and validation data ---
    train_set = NeuronalStructuresAugmented(train=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    val_set = NeuronalStructuresAugmented(subdir_features='val/images', subdir_labels='val/masks', landmarks='landmarks_val.csv', train=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    train_val_set = NeuronalStructuresAugmented(train=True)
    train_val_set.train = False
    train_val_set = Subset(train_val_set, torch.randperm(len(val_set)))
    train_val_loader = DataLoader(train_val_set, BATCH_SIZE, shuffle=False)

    # --- Load test data ---
    test_transform = TestTransform()
    test_set = NeuronalStructuresOriginal(features='test-volume.tif', labels='test-labels.tif', transform=test_transform)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False)

    # --- For data for plotting (see function save_images) ---
    ORIG_TRAIN_SET = NeuronalStructuresOriginal(transform=test_transform)

    # --- Initialize model ---
    unet = UNet(in_channels=1, out_channels=2).to(DEVICE)
    criterion = CrossEntropyLossWithWeights().to(DEVICE)
    optimizer = torch.optim.SGD(unet.parameters(), LEARNING_RATE, MOMENTUM)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    # --- Training ---
    train(unet, train_loader, val_loader, optimizer, criterion, scheduler)
    
    # --- Evaluation ---
    test_loss = evaluate(unet, test_loader, criterion)
    print(f'Test loss: {test_loss}')

    # --- Save trained model ---
    torch.save(unet.state_dict(), SAVE_MODEL_PATH)