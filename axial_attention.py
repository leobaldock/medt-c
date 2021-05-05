##########
# Title: The AI Summer
# Author: Adaloglou Nikolas & Sergios Karagiannakos
# Date: Feb 12 2021
# Code version: 1c9bf6c
# Available: https://github.com/The-AI-Summer/self-attention-cv/blob/main/self_attention_cv/axial_attention_deeplab/axial_attention.py
##########

import torch
from einops import rearrange
from torch import nn

def _conv1d1x1(in_channels, out_channels):
    """1D convolution with kernel size of 1 followed by batch norm"""
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                         nn.BatchNorm1d(out_channels))

class Relative2DPosEncQKV(nn.Module):
    def __init__(self, dim_head, dim_v=16, dim_kq=8):
        """
        Implementation of 2D relative positional embeddings for q,v,k
        Out shape shape will be [dim_head, dim, dim]
        Embeddings are shared across heads for all q,k,v
        Based on Axial DeepLab https://arxiv.org/abs/2003.07853
        Args:
            dim_head: the dimension of the head
            dim_v: d_out in the paper
            dim_kq: d_k in the paper
        """
        super().__init__()
        self.dim = dim_head
        self.dim_head_v = dim_v
        self.dim_head_kq = dim_kq
        self.qkv_chan = 2 * self.dim_head_kq + self.dim_head_v

        # 2D relative position embeddings of q,k,v:
        self.relative = nn.Parameter(torch.randn(self.qkv_chan, dim_head * 2 - 1), requires_grad=True)
        self.relative_index_2d = self.relative_index()

    def relative_index(self):
        # integer lists from 0 to 63
        query_index = torch.arange(self.dim).unsqueeze(0)  # [1, dim]
        key_index = torch.arange(self.dim).unsqueeze(1)  # [dim, 1]

        relative_index_2d = (key_index - query_index) + self.dim - 1  # dim X dim
        return rearrange(relative_index_2d, 'i j->(i j)')  # flatten

    def forward(self):
        rel_indx = self.relative_index_2d.to(self.relative.device)
        all_embeddings = torch.index_select(self.relative, 1, rel_indx)  # [head_planes , (dim*dim)]

        all_embeddings = rearrange(all_embeddings, ' c (x y)  -> c x y', x=self.dim)

        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.dim_head_kq, self.dim_head_kq, self.dim_head_v],
                                                            dim=0)
        return q_embedding, k_embedding, v_embedding


class AxialAttention(nn.Module):
    def __init__(self, dim, in_channels=128, heads=8, dim_head_kq=8, is_gated=False):
        """
        Fig.1 page 6 in Axial DeepLab paper
        Args:
            in_channels: the channels of the feature map to be convolved by 1x1 1D conv
            heads: number of heads
            dim_head_kq: inner dim
        """
        super().__init__()
        self.dim_head = in_channels // heads
        self.dim = dim

        self.heads = heads

        self.dim_head_v = self.dim_head  # d_out
        self.dim_head_kq = dim_head_kq
        self.qkv_channels = self.dim_head_v + self.dim_head_kq * 2
        self.to_qvk = _conv1d1x1(in_channels, self.heads * self.qkv_channels)

        # CODE ADDED TO ORIGINAL
        self.is_gated = is_gated
        if self.is_gated:
            # Implement gate parameters.
            self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False) 
            self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
            self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
            self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)

        # END CODE ADDED

        # Position embedding 2D
        self.RelativePosEncQKV = Relative2DPosEncQKV(dim, self.dim_head_v, self.dim_head_kq)

        # Batch normalization - not common, but we dont need to scale down the dot products this way
        self.attention_norm = nn.BatchNorm2d(heads * 3)
        self.out_norm = nn.BatchNorm1d(in_channels * 2)

    def forward(self, x_in):
        assert x_in.dim() == 3, 'Ensure your input is 3D: [b * width, chan, height] or [b * height, chan, width]'
        # Calculate position embedding -> [ batch*width , qkv_channels,  dim ]
        qkv = self.to_qvk(x_in)

        qkv = rearrange(qkv, 'b (q h) d -> b h q d ', d=self.dim, q=self.qkv_channels, h=self.heads)

        # dim_head_kq != dim_head_v so I cannot decompose with einops here I think
        q, k, v = torch.split(qkv, [self.dim_head_kq, self.dim_head_kq, self.dim_head_v], dim=2)

        r_q, r_k, r_v = self.RelativePosEncQKV()

        # Computations are carried as Fig.1 page 6 in Axial DeepLab paper
        qr = torch.einsum('b h i d, i d j -> b h d j ', q, r_q)
        kr = torch.einsum('b h i d, i d j -> b h d j ', k, r_k)

        # CODE ADDED
        if self.is_gated:
            qr = torch.mul(qr, self.f_qr) # GQ
            kr = torch.mul(kr, self.f_kr) # GK
        # END CODE ADDED

        dots = torch.einsum('b h i d, b h i j -> b h d j', q, k)

        # We normalize the 3 tensors qr, kr, dots together before element-wise addition
        # To do so we concatenate the tensor heads just to normalize them
        # conceptually similar to scaled dot product in MHSA
        # Here n = len(list)
        norm_dots = self.attention_norm(rearrange(list([qr, kr, dots]), 'n b h d j -> b (h n) d j'))

        # Now we can decompose them
        norm_dots = rearrange(norm_dots, 'b (h n) d j -> n b h d j', n=3)

        # And use einsum in the n=3 axis for element-wise sum
        norm_dots = torch.einsum('n b h d j -> b h d j', norm_dots)

        # Last dimension is used softmax and matrix multplication
        attn = torch.softmax(norm_dots, dim=-1)
        # Matrix multiplication will be performed in the dimension of the softmax! Attention :)
        out = torch.einsum('b h d j,  b h i j -> b h i d', attn, v)

        # Last embedding of v
        kv = torch.einsum('b h d j, i d j -> b h i d ', attn, r_v)

        # CODE ADDED
        if self.is_gated:
            kv = torch.mul(kv, self.f_sve) # GV1
            out = torch.mul(out, self.f_sv) # GV2
        # END CODE ADDED

        # To perform batch norm as described in paper,
        # we will merge the dimensions that are != self.dim
        # n = 2 = len(list)
        out = self.out_norm(rearrange(list([kv, out]), 'n b h i d ->  b (n h i ) d'))
        # decompose back output and merge heads
        out = rearrange(out, 'b (n h i ) d ->  n b (h i) d ', n=2, h=self.heads)
        # element wise sum in n=2 axis
        return torch.einsum('n b j i -> b j i', out)