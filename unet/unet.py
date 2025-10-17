import torch
import torch.nn as nn

import torchvision.transforms.functional as TF


class DoubleConv2d(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
            kernel_size: int=3,
            stride: int=1,
            padding: int=0,
        ) -> None:
        super().__init__()        
        self.double_conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv2d(x)


class Down(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int
        ) -> None:
        super().__init__()        
        self.double_conv2d = DoubleConv2d(in_channels, out_channels)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.double_conv2d(x) 
        return self.max_pool2d(x), x


class Up(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int
        ) -> None:
        super().__init__()        
        self.conv2d_up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, padding=0, stride=2)
        self.double_conv2d = DoubleConv2d(in_channels=2*out_channels, out_channels=out_channels)

    def forward(self, x_residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_up(x)        
        _, _, H, W = x.shape 
        x_residual = TF.center_crop(x_residual, output_size=(H, W))
        x = torch.cat([x_residual, x], dim=1) 
        return self.double_conv2d(x)        


class UNet(nn.Module):
    """
    Implementation of the U-Net architecture for image segmentation.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation; Brox et al., 2015
    https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            feature_channel: int=64
        ) -> None:
        super().__init__()        

        # --- Contracting path ---
        self.down1 = Down(in_channels=in_channels, out_channels=feature_channel)
        self.down2 = Down(in_channels=feature_channel, out_channels=2*feature_channel)
        self.down3 = Down(in_channels=2*feature_channel, out_channels=4*feature_channel)
        self.down4 = Down(in_channels=4*feature_channel, out_channels=8*feature_channel)
        # self.dropout = nn.Dropout(0.5)

        # --- Bottleneck ---
        self.bottleneck = DoubleConv2d(in_channels=8*feature_channel, out_channels=16*feature_channel)
        self.dropout = nn.Dropout(0.5)

        # --- Expansive path ---
        self.up1 = Up(in_channels=16*feature_channel, out_channels=8*feature_channel)
        self.up2 = Up(in_channels=8*feature_channel, out_channels=4*feature_channel)
        self.up3 = Up(in_channels=4*feature_channel, out_channels=2*feature_channel)
        self.up4 = Up(in_channels=2*feature_channel, out_channels=feature_channel)
        self.final_conv2d = nn.Conv2d(in_channels=feature_channel, out_channels=out_channels, kernel_size=1, padding=0)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Downwards --- 
        x1_out, x1 = self.down1(x)
        x2_out, x2 = self.down2(x1_out) 
        x3_out, x3 = self.down3(x2_out) 
        x4_out, x4 = self.down4(x3_out) 
        # x4_out = self.dropout(x4_out)

        # --- Bottleneck --- 
        bottleneck_out = self.bottleneck(x4_out)
        bottleneck_out = self.dropout(bottleneck_out)

        # --- Upwards --- 
        up = self.up1(x4, bottleneck_out)
        up = self.up2(x3, up)
        up = self.up3(x2, up)
        up = self.up4(x1, up)
        return self.final_conv2d(up)
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)