import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageSequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision.transforms import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class SegmentationTransform:
    def __init__(self, to_tensor: bool=True) -> None:
        self.to_tensor = to_tensor
    
    def __call__(self, img: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TrainTransform(SegmentationTransform):
    """
    Applies data augmentation to input and target data.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """ 
    def __call__(
            self, 
            img: Image.Image, 
            mask: Image.Image, 
            img_out_shape: list=[572, 572], 
            mask_out_shape: list=[388, 388]
        ) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Resize ---
        img = TF.resize(img, size=mask_out_shape, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size=mask_out_shape, interpolation=TF.InterpolationMode.NEAREST)

        mask = TF.pad(mask, padding=4*92, padding_mode='reflect')
        img = TF.pad(img, padding=4*92, padding_mode='reflect')

        # --- Elastic transform ---
        H, W = mask.size
        elastic = T.ElasticTransform(alpha=12.0, sigma=10.0)
        disp = elastic.get_params(elastic.alpha, elastic.sigma, size=[H, W])
        img = TF.elastic_transform(img, disp, transforms.InterpolationMode.BILINEAR)
        mask = TF.elastic_transform(mask, disp, transforms.InterpolationMode.NEAREST)

        # --- Random rotation ---
        angle = random.uniform(0.0, 180.0)
        img = TF.rotate(img, angle, transforms.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, transforms.InterpolationMode.NEAREST)

        # --- Random shift transformation ---
        tx = int(random.uniform(-0.3, 0.3) * mask_out_shape[0])
        ty = int(random.uniform(-0.3, 0.3) * mask_out_shape[1])
        scale = random.uniform(0.97, 1.03)
        img = TF.affine(img, 0.0, (tx, ty), scale, 0.0, transforms.InterpolationMode.BILINEAR)
        mask = TF.affine(mask, 0.0, (tx, ty), scale, 0.0, transforms.InterpolationMode.NEAREST)

        # --- Center crop ---
        mask = TF.center_crop(mask, mask_out_shape)
        img = TF.center_crop(img, (img_out_shape[0], img_out_shape[1]))
        
        # --- Random intensity scaling ---
        brightness_factor = random.uniform(0.5, 1.5)
        img = TF.adjust_brightness(img, brightness_factor) 
        
        # --- To tensor ---
        if self.to_tensor: 
            img = TF.to_tensor(img)
            mask = TF.pil_to_tensor(mask).squeeze(0)
            mask = torch.clamp(mask, 0, 1)
        return img, mask    # [1, 572, 572], [388, 388]


class TestTransform(SegmentationTransform):
    """
    Applies data transformation to test data (input and target).
    
    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """ 
    def __call__(
            self, 
            img: Image.Image, 
            mask: Image.Image, 
            mask_out_shape: list=[388, 388]
        ) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Resize --- 
        img = TF.resize(img, size=mask_out_shape, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size=mask_out_shape, interpolation=TF.InterpolationMode.NEAREST)

        # --- Mirroring --- 
        img = TF.pad(img, padding=92, padding_mode='reflect')

        # --- To tensor ---
        if self.to_tensor: 
            img = TF.to_tensor(img)
            mask = TF.pil_to_tensor(mask).squeeze(0)
            mask = torch.clamp(mask, 0, 1)
        return img, mask


class ValidationTransform(SegmentationTransform):
    """
    Applies data augmentation to input and target data.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """ 
    def __call__(
            self, 
            img: Image.Image, 
            mask: Image.Image, 
            img_out_shape: list=[572, 572], 
            mask_out_shape: list=[388, 388]
        ) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Resize ---
        img = TF.resize(img, size=mask_out_shape, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size=mask_out_shape, interpolation=TF.InterpolationMode.NEAREST)

        mask = TF.pad(mask, padding=4*92, padding_mode='reflect')
        img = TF.pad(img, padding=4*92, padding_mode='reflect')

        # --- Random rotation ---
        angle = random.uniform(0.0, 180.0)
        img = TF.rotate(img, angle, transforms.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, transforms.InterpolationMode.NEAREST)

        # --- Random shift transformation ---
        tx = int(random.uniform(-0.3, 0.3) * mask_out_shape[0])
        ty = int(random.uniform(-0.3, 0.3) * mask_out_shape[1])
        scale = random.uniform(0.97, 1.03)
        img = TF.affine(img, 0.0, (tx, ty), scale, 0.0, transforms.InterpolationMode.BILINEAR)
        mask = TF.affine(mask, 0.0, (tx, ty), scale, 0.0, transforms.InterpolationMode.NEAREST)

        # --- Center crop ---
        mask = TF.center_crop(mask, mask_out_shape)
        img = TF.center_crop(img, (img_out_shape[0], img_out_shape[1]))
        
        # --- To tensor ---
        if self.to_tensor: 
            img = TF.to_tensor(img)
            mask = TF.pil_to_tensor(mask).squeeze(0)
            mask = torch.clamp(mask, 0, 1)
        return img, mask    # [1, 572, 572], [388, 388]


class NeuronalStructuresOriginal(Dataset):
    """
    Neuronal Structures in EM Stacks Dataset.

    Reference:
    ISBI 2012 challange: Segmentation of neuronal structures in EM stacks.
    https://imagej.net/events/isbi-2012-segmentation-challenge 
    
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """

    def __init__(
            self,
            root: str='./ISBI-2012-challenge/',
            features: str='train-volume.tif',
            labels: str='train-labels.tif',
            transform: None | SegmentationTransform = None,
        ) -> None:
        self.root = root
        self.transform = transform
        self.data = [image.copy() for image in ImageSequence.Iterator(Image.open(root + features))]
        self.targets = [image.copy() for image in ImageSequence.Iterator(Image.open(root + labels))]
    
    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.item()

        img, mask = self.data[idx], self.targets[idx]
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask


class NeuronalStructuresAugmented(Dataset):
    """
    Augmented version of the Neuronal Structures in EM Stacks dataset.
    This verison is used for training purposes only.

    Expected folder structure:
    isbi-2012/    
        /train/
            /images/
            /masks/
        /val/
            /images
            /masks/

    Reference:
    ISBI 2012 challange: Segmentation of neuronal structures in EM stacks.
    https://imagej.net/events/isbi-2012-segmentation-challenge 
    
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """
    def __init__(
            self,
            root: str='../../datasets/isbi-2012/',
            subdir_features: str='train/images',
            subdir_labels: str='train/masks',
            subdir_weights: None|str='train/weights',
            landmarks: str='landmarks_train.csv',
            train: bool=True
        ) -> None:
        assert os.path.exists(root), f"Path {root} does not exist"
        self.root = root
        self.path_features = os.path.join(root, subdir_features)
        self.path_labels= os.path.join(root, subdir_labels)
        self.path_weights = os.path.join(root, subdir_weights)

        assert os.path.exists(self.path_features), f"Path {self.path_features} does not exist"
        assert os.path.exists(self.path_labels), f"Path {self.path_labels} does not exist"

        path_landmarks = os.path.join(self.root, landmarks) 
        assert os.path.exists(path_landmarks), f"Path {path_landmarks} does not exist"

        self.df_landmarks = pd.read_csv(path_landmarks) 
        self.train = train

    def __len__(self) -> int:
        return len(self.df_landmarks)

    def __getitem__(self, idx: int | torch.Tensor) -> tuple:
        if torch.is_tensor(idx):
            idx = idx.item()
        img_path = os.path.join(self.path_features, self.df_landmarks.iloc[idx, 0])
        mask_path = os.path.join(self.path_labels, self.df_landmarks.iloc[idx, 1])

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        img = TF.to_tensor(img)
        mask = TF.pil_to_tensor(mask).squeeze(0)
        mask = torch.clamp(mask, 0, 1)
        
        if self.train:
            weight_path = os.path.join(self.path_weights, self.df_landmarks.iloc[idx, 2])
            weight = np.load(weight_path)
            weight = torch.from_numpy(weight)
            return img, mask, weight
        
        return img, mask