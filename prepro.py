import os
import random
import argparse
from argparse import Namespace

import numpy as np
import pandas as pd

from scipy import ndimage
from scipy.ndimage import distance_transform_edt

from PIL import Image, ImageSequence

from sklearn.utils.class_weight import compute_class_weight

import torch

from unet.dataset import TrainTransform, ValidationTransform


def compute_weight_map(targets: np.ndarray) -> np.ndarray:
    targets_flat = targets.ravel()
    classes = np.unique(targets_flat)
    weight_map = compute_class_weight(class_weight='balanced', classes=classes, y=targets_flat)
    return weight_map


def compute_distance_map(target: np.ndarray, w0: float=10.0, sigma: float=5.0) -> np.ndarray:
    cells = (target == 1)
    eroded = ndimage.binary_erosion(cells, border_value=1)
    boundary = np.logical_xor(cells, eroded)
    d1 = distance_transform_edt(target)
    d2 = distance_transform_edt(1 - boundary)
    weight_dist = w0 * np.exp(-((d1 + d2)**2)/(2*sigma**2))
    return weight_dist


def set_seed(seed: int = 42) -> None:
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)


def prepro_isbi_2012_dataset(
        root: str='./ISBI-2012-challenge/',
        features: str='train-volume.tif',
        labels: str='train-labels.tif',
        path_save_images: str='../../datasets/isbi-2012/',
        num_train_samples: int=60000,
        num_val_samples: int=6000,
        w0: float=10.0,
        sigma: float=5.0,
        seed: int=42,
        verbose: bool=True
    ) -> None:
    """Creates augmented training and validation data for the ISBI-2012 neuronal segmentation dataset.""" 
    assert os.path.exists(root), "ISBI-2012 data not found."
    path = path_save_images
    path_train = os.path.join(path, './train/')
    path_val = os.path.join(path, './val/')

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path_train):
        os.mkdir(path_train)
        os.mkdir(os.path.join(path_train, './images/'))
        os.mkdir(os.path.join(path_train, './masks/'))
        os.mkdir(os.path.join(path_train, './weights/'))
    if not os.path.exists(path_val):
        os.mkdir(path_val)
        os.mkdir(os.path.join(path_val, './images/'))
        os.mkdir(os.path.join(path_val, './masks/'))

    # --- Loading the data ---
    data = [image.copy() for image in ImageSequence.Iterator(Image.open(root + features))]
    targets = [image.copy() for image in ImageSequence.Iterator(Image.open(root + labels))]
    
    transform_train = TrainTransform(to_tensor=False)
    transform_val = ValidationTransform(to_tensor=False)

    # --- Computing the data splits ---
    set_seed(seed)
    indices = np.random.permutation(len(data))
    train_size = int(0.8 * len(data))
    indices_train = indices[:train_size] 
    indices_val = indices[train_size:]

    # --- Pre-compute class weight ---
    ys = []
    for y in targets:
        y = np.array(y) 
        ys.append(y)
    ys = np.concatenate(ys, axis=0)
    weight_c = compute_weight_map(ys)

    # --- Training data ---
    N_aug_imgs = 0
    img_names = [] 
    mask_names = [] 
    weight_names = [] 
    while N_aug_imgs < num_train_samples: 
        perm = np.random.permutation(train_size) 
        for j in indices_train[perm]:
            img, mask = data[j], targets[j]
            img, mask = transform_train(img, mask)

            # --- Compute the weights --- 
            mask_np = np.array(mask.copy()).clip(0, 1)
            weight_d = compute_distance_map(mask_np, w0, sigma)
            weight = np.take(weight_c, mask_np)
            weight = weight + weight_d

            # --- Save augmented data ---
            img.save(os.path.join(path_train, f'./images/img_train_{N_aug_imgs}.png'))
            mask.save(os.path.join(path_train, f'./masks/mask_train_{N_aug_imgs}.png'))
            np.save(os.path.join(path_train, f'./weights/weight_train_{N_aug_imgs}.npy'), weight)
            
            img_names.append(f'img_train_{N_aug_imgs}.png')
            mask_names.append(f'mask_train_{N_aug_imgs}.png')
            weight_names.append(f'weight_train_{N_aug_imgs}.npy')
            N_aug_imgs += 1

            if verbose and N_aug_imgs % 1000 == 0:
                print(f'Wrote {N_aug_imgs}/{num_train_samples} augmented training images') 
    df = pd.DataFrame({'images': img_names, 'masks': mask_names, 'weights': weight_names})
    df.to_csv(os.path.join(path, 'landmarks_train.csv'), index=False)

    # --- Validation data ---
    N_aug_imgs = 0
    img_names = []
    mask_names = []
    while N_aug_imgs < num_val_samples: 
        perm = np.random.permutation(len(indices_val)) 
        for j in indices_val[perm]:
            img, mask = data[j], targets[j]
            img, mask = transform_val(img, mask)
            
            # --- Save augmented data ---
            img.save(os.path.join(path_val, f'./images/img_val_{N_aug_imgs}.png'))
            mask.save(os.path.join(path_val, f'./masks/mask_val_{N_aug_imgs}.png'))
            img_names.append(f'img_val_{N_aug_imgs}.png')
            mask_names.append(f'mask_val_{N_aug_imgs}.png')
            N_aug_imgs += 1
            
            if verbose and N_aug_imgs % 1000 == 0:
                print(f'Wrote {N_aug_imgs}/{num_train_samples} augmented validation images') 
    df = pd.DataFrame({'images': img_names, 'masks': mask_names})
    df.to_csv(os.path.join(path, 'landmarks_val.csv'), index=False)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog='Create a augmented training and validation set for UNet training',
        description='Train a UNet model on the ISBI-2012 dataset for neuronal structure segmentation.'
    )
    parser.add_argument('--num_train_samples', type=int, default=60000)
    parser.add_argument('--num_val_samples', type=int, default=6000)
    parser.add_argument('--w0', type=float, default=10.0)
    parser.add_argument('--sigma', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    prepro_isbi_2012_dataset(
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        w0=args.w0,
        sigma=args.sigma,
        seed=args.seed,
        verbose=args.verbose
    )