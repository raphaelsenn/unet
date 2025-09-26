import random

from PIL import Image, ImageSequence

import torch
from torch.utils.data import Dataset

from torchvision.transforms import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class SegmentationTransform:
    def __call__(self, img: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TrainTransform(SegmentationTransform):
    """
    Applies data augmentation to input and target data.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """ 
    def __call__(self, img: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        W, H = img.size
        
        # Elastic transform
        elastic = T.ElasticTransform(alpha=50.0, sigma=5.0)
        disp = elastic.get_params(elastic.alpha, elastic.sigma, size=[H, W])
        img = TF.elastic_transform(img, displacement=disp, fill=0)
        mask = TF.elastic_transform(mask, displacement=disp, fill=0)

        # Random rotation
        angle = random.uniform(0.0, 180.0)
        img = TF.rotate(img, angle, transforms.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, transforms.InterpolationMode.BILINEAR)

        # Random affine transformation
        degrees = random.uniform(30.0, 70.0) 
        tx = int(random.uniform(0.1, 0.3) * W)
        ty = int(random.uniform(0.1, 0.3) * H)
        scale = random.uniform(0.5, 0.75)
        shear = 0.0
        img = TF.affine(img, degrees, (tx, ty), scale, shear, transforms.InterpolationMode.BILINEAR, fill=0)
        mask = TF.affine(mask, degrees, (tx, ty), scale, shear, transforms.InterpolationMode.BILINEAR, fill=0)

        # Random intensity scaling
        brightness_factor = random.uniform(0.5, 1.5)
        img = TF.adjust_brightness(img, brightness_factor) 

        # To tensors
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=0.5, std=0.5)
        mask = TF.pil_to_tensor(mask).squeeze(0)

        return img, mask


class TestTransform(SegmentationTransform):
    """
    Applies data transformation to test data (input and target).
    
    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """ 
    def __call__(self, img: Image, mask: Image) -> tuple[torch.Tensor, torch.Tensor]:
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=0.5, std=0.5)
        mask = TF.pil_to_tensor(mask).squeeze(0)
        return img, mask


class NeuronalStructures(Dataset):
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