import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.transforms import CenterCrop

from unet.unet import UNet
from unet.dataset import NeuronalStructures, TrainTransform, TestTransform


def train(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim,
        criterion: nn.Module,
    ) -> None:
    model.train() 
    for epoch in range(EPOCHS): 
        total_loss = 0.0 
        for x, y in dataloader:
            N_batch = x.shape[0]
            x, y = x.to(DEVICE), y.to(DEVICE)

            x = F.pad(x, pad=(PADDING, PADDING, PADDING, PADDING), mode=PADDING_MODE)
            y_pred = model(x)
            y_pred = CenterCrop(size=y.shape[-2:])(y_pred)
            
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * N_batch 
        if VERBOSE: 
            total_loss /= len(dataloader.dataset) 
            print(f'epoch: {epoch+1}/{EPOCHS}\tloss: {total_loss:.4f}')


@torch.no_grad()
def evaluate(model, dataloader, criterion) -> float:
    model.eval() 
    total_loss = 0.0 
    for x, y in dataloader:
        N_batch = x.shape[0]
        x, y = x.to(DEVICE), y.to(DEVICE)

        x = F.pad(x, pad=(PADDING, PADDING, PADDING, PADDING), mode=PADDING_MODE)
        y_pred = model(x)
        y_pred = CenterCrop(size=y.shape[-2:])(y_pred)

        loss = criterion(y_pred, y)
        total_loss += loss.item() * N_batch
    return total_loss


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # --- Settings and Hyperparameters ---
    EPOCHS = 5
    BATCH_SIZE = 1
    HYPER = {'lr': 0.07450565708394236, 'momentum': 0.8763924230737281}
    PADDING = 96
    PADDING_MODE = 'reflect'
    VERBOSE = True
    
    SAVE_MODEL_PATH = './unet_trained.pth'
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    set_seed(SEED)
    
    # --- Loading training and testing data ---
    train_transform = TrainTransform()
    train_set = NeuronalStructures(transform=train_transform)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    
    test_transform = TestTransform()
    test_set = NeuronalStructures(features='test-volume.tif', labels='test-labels.tif', transform=test_transform)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False)

    # --- Initialize model ---
    unet = UNet(in_channels=1, out_channels=2).to(DEVICE)
    optimizer = torch.optim.SGD(unet.parameters(), HYPER['lr'], HYPER['momentum'])
    criterion = nn.CrossEntropyLoss()

    # --- Training ---
    train(unet, train_loader, optimizer, criterion)
    
    # --- Evaluation ---
    test_loss = evaluate(unet, test_loader, criterion)
    print(f'Test loss: {test_loss}')

    # --- Save trained model ---
    torch.save(unet.state_dict(), SAVE_MODEL_PATH)

