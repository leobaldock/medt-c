#!/usr/bin/env python
# coding: utf-8
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import torch.autograd.profiler as profiler

from axial_attention import AxialAttention as PositionSensitiveAxialAttention


class AxialAttention(nn.Module):
    def __init__(self, embed_size, heads, is_width=False):
        super(AxialAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.is_width = is_width

        assert (self.head_dim * heads ==
                embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, x):
        if self.is_width:
            y = x.permute(0, 2, 3, 1)
        else:
            y = x.permute(0, 3, 2, 1)
        N, D1, D2, C = y.shape
        y = y.contiguous().view(N * D1, D2, C)

        # Split input into self.heads chunks. possible that this needs to go after the linear projections
        y = y.reshape(
            N * D1,
            D2,
            self.heads,
            self.head_dim
        )

        values = self.values(y)
        keys = self.keys(y)
        queries = self.queries(y)

        # this is the QK matrix multiply step
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # concat the heads with wierd reshape thing
        y = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, D1, D2, C
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum: (N, query_len, heads, head_dim) then flatten last two dimensions

        y = self.fc_out(y)
        if self.is_width:
            y = y.permute(0, 3, 1, 2)
        else:
            y = y.permute(0, 3, 2, 1)
        return y


class GatedAxialTransformerLayer(nn.Module):
    def __init__(self, in_channels, img_dim, heads):
        super(GatedAxialTransformerLayer, self).__init__()
        self.in_channels = in_channels
        self.attention_channels = in_channels // 2  # downsample by 50%
        self.img_dim = img_dim
        self.heads = heads

        # Conv 1x1 to halve size
        self.conv_down = nn.Conv2d(
            self.in_channels, self.attention_channels, kernel_size=1, stride=1)

        # Norm
        self.bn1 = nn.BatchNorm2d(self.attention_channels)

        # GatedAxialAttention height
        self.height_attention = PositionSensitiveAxialAttention(
            dim=img_dim, in_channels=self.attention_channels, heads=self.heads,
            is_gated=False
        )

        # GatedAxialAttention width
        self.width_attention = PositionSensitiveAxialAttention(
            dim=img_dim, in_channels=self.attention_channels, heads=self.heads,
            is_gated=False
        )

        # Conv 1x1 to increase size
        self.conv_up = nn.Conv2d(
            self.attention_channels, in_channels, kernel_size=1, stride=1)

        # Norm
        self.bn2 = nn.BatchNorm2d(self.in_channels)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert x.dim() == 4, 'Ensure input is [batch, channels, height, width]'

        # Conv 1x1 to halve size
        y = self.conv_down(x)

        # Norm
        y = self.bn1(y)

        # ReLU
        y = self.relu(y)

        # Merge batch dimension with width
        y = rearrange(y, 'b c h w -> (b w) c h')
        # GatedAxialAttention height
        y = self.height_attention(y)

        # Decompose width and merge batch with height
        y = rearrange(y, '(b w) c h  -> (b h) c w', w=self.img_dim)
        # GatedAxialAttention width
        y = self.width_attention(y)
        # Decompose to original dimensions
        y = rearrange(y, '(b h) c w -> b c h w', h=self.img_dim)

        # ReLU
        y = self.relu(y)

        # Conv 1x1 to increase size
        y = self.conv_up(y)

        # Norm
        y = self.bn2(y)

        # Residual connection addition from input
        y += x

        # ReLU
        y = self.relu(y)

        return y


class Branch(nn.Module):
    def __init__(
        self,
        img_dim,
        feature_channels,
        num_transformer_layers,
        color_channels=3,
        conv_kernel_size=7,
        heads=8,
    ):
        super(Branch, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=color_channels,
                out_channels=feature_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(
            #     in_channels=feature_channels,
            #     out_channels=feature_channels * 2,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     bias=False
            # ),
            # nn.BatchNorm2d(feature_channels * 2),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(
            #     in_channels=feature_channels * 2,
            #     out_channels=feature_channels,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     bias=False

            # ),
            # nn.BatchNorm2d(feature_channels),
            # nn.ReLU(inplace=True)
        )

        self.transformer = nn.ModuleList(
            [
                # Transformer.
                GatedAxialTransformerLayer(
                    in_channels=feature_channels, img_dim=img_dim, heads=heads)
                for _ in range(num_transformer_layers)
            ],
        )

    def forward(self, x):
        y = self.cnn(x)

        for layer in self.transformer:
            y = layer(y)

        return y


class Encoder(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        feature_dim,
    ):
        super(Encoder, self).__init__()

        assert img_dim % patch_dim == 0, f"Image dimension {img_dim} is not divisible by patch dimension {patch_dim}."
        self.img_dim = img_dim
        self.patch_dim = patch_dim

        self.global_branch = Branch(img_dim, feature_dim, 2)
        self.local_branch = Branch(patch_dim, feature_dim, 5)

    def forward(self, x):
        yg = self.global_branch(x)
        # could probably do this with convolution and then reshape, would likely be faster...
        yl = yg.clone()
        for i in range(0, self.img_dim // self.patch_dim):
            for j in range(0, self.img_dim // self.patch_dim):
                patch = x[:, :, self.patch_dim*i:self.patch_dim *
                          (i+1), self.patch_dim*j:self.patch_dim*(j+1)]
                y_patch = self.local_branch(patch)
                yl[:, :, self.patch_dim*i:self.patch_dim *
                    (i+1), self.patch_dim*j:self.patch_dim*(j+1)] = y_patch

        y = yg + yl  # not sure if this is the right way to do a summation?
        return y


class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Decoder, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        y = self.avg(x)
        y = y.view(y.size(0), -1)
        y = self.decoder(y)
        return y


class MedT_C(nn.Module):
    def __init__(
        self,
        img_dim=32,
        in_channels=3,
        patch_dim=8,
        num_classes=10,
        feature_dim=256,
    ):
        super(MedT_C, self).__init__()

        self.encoder = Encoder(img_dim, patch_dim, feature_dim)
        self.decoder = Decoder(feature_dim, num_classes)

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y
