from math import log, sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import FastAttention as _FastAttention
from utils.masking import ProbMask, TriangularCausalMask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=False,
        scale=None,
        attention_dropout=0.1,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = utils.masking.TriangularCausalMask(B, L, device=queries.device)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # expand heads dimension
            scores.masked_fill_(attn_mask, -np.inf)

        A = torch.nan_to_num(self.dropout(torch.softmax(scale * scores, dim=-1)))
        # A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if output_attn:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class PerformerAttention(_FastAttention):
    def __init__(
        self,
        mask_flag=False,
        dim_heads=None,
        ortho_scaling=0,
        feature_redraw_interval=1000,
        kernel="softmax",
    ):
        assert dim_heads is not None
        super().__init__(
            dim_heads=dim_heads,
            ortho_scaling=ortho_scaling,
            nb_features=max(100, int(dim_heads * log(dim_heads))),
            causal=mask_flag,
            generalized_attention=kernel == "relu",
            kernel_fn=nn.ReLU() if kernel == "relu" else "N/A",
        )
        self.redraw_interval = feature_redraw_interval
        self.register_buffer("calls_since_last_redraw", torch.tensor(0))

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        if self.training:
            if self.calls_since_last_redraw >= self.redraw_interval:
                self.redraw_projection_matrix(queries.device)
                self.calls_since_last_redraw.zero_()
            else:
                self.calls_since_last_redraw += 1

        queries = queries.transpose(1, 2).bfloat16()
        keys = keys.transpose(1, 2).bfloat16()
        values = values.transpose(1, 2).bfloat16()
        v = super().forward(queries, keys, values)

        return v.transpose(1, 2), None


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_queries_keys,
        d_values,
        n_heads,
        dropout_qkv=0.0,
        mix=False,
    ):
        super(AttentionLayer, self).__init__()

        self.inner_attention = attention()
        self.query_projection = nn.Linear(d_model, d_queries_keys * n_heads).bfloat16()
        self.key_projection = nn.Linear(d_model, d_queries_keys * n_heads).bfloat16()
        self.value_projection = nn.Linear(d_model, d_values * n_heads).bfloat16()
        self.out_projection = nn.Linear(d_values * n_heads, d_model).bfloat16()
        self.dropout_qkv = nn.Dropout(dropout_qkv).bfloat16()
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.dropout_qkv(self.query_projection(queries)).view(B, L, H, -1).bfloat16()
        keys = self.dropout_qkv(self.key_projection(keys)).view(B, S, H, -1).bfloat16()
        values = self.dropout_qkv(self.value_projection(values)).view(B, S, H, -1).bfloat16()

        out, attn = self.inner_attention(
            queries=queries,
            keys=keys,
            values=values,
            attn_mask=attn_mask,
            output_attn=output_attn,
        )

        if output_attn and attn is None:
            onehot_values = (
                torch.eye(S).unsqueeze(0).repeat(B, 1, 1).unsqueeze(2).to(values.device)
            )
            with torch.no_grad():
                attn, _ = self.inner_attention(
                    queries=queries,
                    keys=keys,
                    values=onehot_values,
                    attn_mask=attn_mask,
                )
                attn = attn.transpose(2, 1)

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        if not output_attn:
            assert attn is None

        out = self.out_projection(out)
        return out, attn
