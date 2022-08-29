#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This file implements the Mutual Attention Transformer (SGA-SGA)

"""

import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional, Any, Union, Callable


class MutualAttentionTransformer(nn.Module):
    """
    Implementing the Mutual Attention Transformer (SGA-SGA) approach
    Deep Features Fusion with Mutual Attention Transformer for Skin Lesion Diagnosis - https://ieeexplore.ieee.org/document/9506211
    """

    def __init__(self, d_model, hidden_size=512, num_heads=8, dropout=0.1, batch_first=True):
        super(MutualAttentionTransformer, self).__init__()

        self.transformer_unit = SGASGA(d_model,
                                       hidden_size=4 * d_model,
                                       num_heads=8,
                                       dropout=0.1)
        self.fusion_unit = MATFusion(d_model, dropout=0.1)

        self.batch_first = batch_first

    def forward(self, x, y):
        if self.batch_first:
            x = x.permute(1, 0, 2)
            y = y.permute(1, 0, 2)
        x_hat, y_hat = self.transformer_unit(x, y)
        z = self.fusion_unit(x_hat, y_hat)
        return z


class SGASGA(nn.Module):
    def __init__(self, d_model, hidden_size=512, num_heads=8, dropout=0.1):
        super(SGASGA, self).__init__()

        self.sa1 = SelfAttentionBlock(d_model, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.ga1 = GuidedAttentionBlock(d_model, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)

        self.sa2 = SelfAttentionBlock(d_model, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.ga2 = GuidedAttentionBlock(d_model, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)

    def forward(self, x, y):
        x, x_int = self.sa1(x)
        y, y_int = self.sa2(y)

        x = self.ga1(x, y_int)
        y = self.ga2(y, x_int)

        return x, y


# def _get_clones(module, N):
#     return ModuleList([copy.deepcopy(module) for i in range(N)])
#

# def _get_activation_fn(activation):
#     if activation == "relu":
#         return nn.F.relu
#     elif activation == "gelu":
#         return nn.F.gelu
#
#     raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation=nn.functional.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None):
        super(SelfAttentionBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout_sa = nn.Dropout(dropout)

        # Implementation of Feedforward model
        self.linear_ff_sa1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ff_sa1 = nn.Dropout(dropout)
        self.linear_ff_sa2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_ff_sa2 = nn.Dropout(dropout)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = activation

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x_intermediate = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._ff_sa_block(self.norm2(x_intermediate))
        else:
            x_intermediate = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._ff_sa_block(x_intermediate))

        return x, x_intermediate

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout_sa(x)

    # feed forward block
    def _ff_sa_block(self, x: Tensor) -> Tensor:
        x = self.linear_ff_sa2(self.dropout_ff_sa1(self.activation(self.linear_ff_sa1(x))))
        return self.dropout_ff_sa2(x)


class GuidedAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation=nn.functional.relu,
                 layer_norm_eps=1e-5, norm_first=False):
        super(GuidedAttentionBlock, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout_mha = nn.Dropout(dropout)

        # Implementation of Feedforward model
        self.linear_ff_ga1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ff_ga1 = nn.Dropout(dropout)
        self.linear_ff_ga2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_ff_ga2 = nn.Dropout(dropout)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = activation

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm1(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_ga_block(self.norm2(x))
        else:
            x = self.norm1(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm2(x + self._ff_ga_block(x))

        return x

    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout_mha(x)

    # feed forward block
    def _ff_ga_block(self, x: Tensor) -> Tensor:
        x = self.linear_ff_ga2(self.dropout_ff_ga1(self.activation(self.linear_ff_ga1(x))))
        return self.dropout_ff_ga2(x)


class MATFusion(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(MATFusion, self).__init__()

        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.functional.relu

    def forward(self, x, y):
        # x, y: n x d_model
        x = self.dropout1(self.activation(self.linear1(x)))  # x: d_model
        # x = torch.flatten(x)

        y = self.dropout2(self.activation(self.linear2(x)))  # y: d_model
        # y = torch.flatten(y)

        z = x * y
        return z  # z: d_model
