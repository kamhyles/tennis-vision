"""TrackNet V1-style PyTorch architecture used by the tennis checkpoint.

The layer contract follows the unofficial PyTorch TrackNet implementation by
yastrebksv, which implements the temporal model described in Huang et al. (2019).
See THIRD_PARTY.md for exact provenance and licensing caveats.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TrackNetV1(nn.Module):
    """Three-frame, 256-class heatmap TrackNet model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBlock(9, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, 256)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.pool1(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.pool2(outputs)
        outputs = self.conv5(outputs)
        outputs = self.conv6(outputs)
        outputs = self.conv7(outputs)
        outputs = self.pool3(outputs)
        outputs = self.conv8(outputs)
        outputs = self.conv9(outputs)
        outputs = self.conv10(outputs)
        outputs = self.ups1(outputs)
        outputs = self.conv11(outputs)
        outputs = self.conv12(outputs)
        outputs = self.conv13(outputs)
        outputs = self.ups2(outputs)
        outputs = self.conv14(outputs)
        outputs = self.conv15(outputs)
        outputs = self.ups3(outputs)
        outputs = self.conv16(outputs)
        outputs = self.conv17(outputs)
        outputs = self.conv18(outputs)
        return outputs.reshape(batch_size, 256, -1)
