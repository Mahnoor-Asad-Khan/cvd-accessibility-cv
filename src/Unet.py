"""
Small U-Net for learned CVD image correction.

Architecture: encoder (downsampling conv blocks) -> bottleneck ->
decoder (upsampling conv blocks with skip connections from the encoder).
Input and output are both (B, 3, H, W) RGB images - the network learns
to output a corrected version of its input, same spatial size.

Trained self-supervised (no labeled dataset) - see train.py for the
loss function, which reuses this project's simulation + accessibility
metric code (in differentiable/torch form) as the training signal.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm + ReLU.
    This is the basic repeated building block used at every U-Net level."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    """
    A small U-Net: 3 encoder levels, 1 bottleneck, 3 decoder levels.
    Filter counts follow the standard "double on downsample" convention:
    16 -> 32 -> 64 -> (bottleneck 128) -> 64 -> 32 -> 16.
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)  # halves spatial size each call

        # Bottleneck
        self.bottleneck = ConvBlock(64, 128)

        # Decoder - ConvTranspose2d upsamples (doubles spatial size)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)  # 128 in = 64 (upsampled) + 64 (skip connection)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)  # 64 = 32 (upsampled) + 32 (skip)

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)  # 32 = 16 (upsampled) + 16 (skip)

        # Final 1x1 conv: map back to 3 output channels (RGB)
        self.output_conv = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        #encoder path
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        #bottleneck
        b = self.bottleneck(p3)

        #decoder path
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        #final output
        return self.output_conv(d1)
        # TODO (Kalon): wire up the forward pass. Pattern:
    