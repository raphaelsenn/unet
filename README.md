# unet
A PyTorch implementation of the **U-Net** architecture for image segmentation.

![unet_architecture](./assets/unet_architecture.png)

*Taken from Brox, Ronneberger, Fischer (2015)*

## Usage

```python
import torch
from unet.unet import UNet


unet = Unet(in_channels=1, out_channels=2)

# input shape [64, 1, 572, 572]
input = torch.rand((64, 1, 572, 572))

# output shape [64, 2, 388, 388]
output = unet(input)
```

## Citations

```bibtex
@misc{ronneberger2015unetconvolutionalnetworksbiomedical,
      title={U-Net: Convolutional Networks for Biomedical Image Segmentation}, 
      author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
      year={2015},
      eprint={1505.04597},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1505.04597}, 
}
```