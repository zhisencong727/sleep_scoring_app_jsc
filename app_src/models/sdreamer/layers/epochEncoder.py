import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from layers.attention import FeedForward
from layers.patchEncoder import PatchEncoder


class mlp_proj(nn.Module):
    # Batch x Epoch x Trace x Channel x Time
    # Batch x Epoch x Trace x Channel x (Patch_num x Patch_len)
    # Batch x Epoch x Trace x Patch_num x (Channel x Patch_len)
    # Batch x Epoch x Trace x d_model

    def __init__(self, in_dim=512, d_model=128, dropout=0.1):
        super().__init__()

        self.to_epoch_embedding = nn.Sequential(
            Rearrange("b ... c t -> b ... (c t)"),
            nn.Linear(in_dim, d_model, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, x):
        x = self.to_epoch_embedding(x)
        return x


class MLP_Block(nn.Module):
    def __init__(self, dim, dropout=0.0, activation="glu"):
        super().__init__()
        glu, relu, relu_squared = False, False, False
        if activation == "glu":
            glu = True
        elif activation == "relu":
            relu = True
        else:
            relu_squared = True
        self.ff = FeedForward(
            dim, mult=1, dropout=dropout, glu=glu, relu=relu, relu_squared=relu_squared
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.ff(self.norm1(x))
        x = self.norm2(x)
        return x


class MLP_Encoder(nn.Module):
    def __init__(self, dim, depth=3, dropout=0.0, activation="glu"):
        super().__init__()
        self.mlp_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mlp_blocks.append(
                MLP_Block(dim, dropout=dropout, activation=activation)
            )

    def forward(self, x):
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)
        return x


class MLP_EpochEncoder(nn.Module):
    def __init__(self, seq_len, dim, depth=3, dropout=0.0, activation="glu"):
        super().__init__()
        self.proj = mlp_proj(seq_len, d_model=dim, dropout=dropout)
        self.mlp_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mlp_blocks.append(
                MLP_Block(dim, dropout=dropout, activation=activation)
            )

    def forward(self, x):
        x = self.proj(x)
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)
        return x


class LSTM_Encoder(nn.Module):
    def __init__(self, dim, depth=3, dropout=0.0, activation="glu"):
        super().__init__()
        self.bilstm = nn.LSTM(
            dim, dim, depth, dropout=dropout, batch_first=True, bidirectional=True
        )
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        x = self.proj(lstm_out)
        return x


class LSTM_EpochEncoder(nn.Module):
    def __init__(self, patch_len, dim, depth=3, dropout=0.0, activation="glu"):
        super().__init__()
        self.patch_enc = PatchEncoder(patch_len, in_channel=1, d_model=dim)
        self.bilstm = nn.LSTM(
            dim, dim, depth, dropout=dropout, batch_first=True, bidirectional=True
        )
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = self.patch_enc(x)
        b, e, t, d = x.shape
        x = rearrange(x, "b e t d -> (b e) t d")
        lstm_out, _ = self.bilstm(x)
        x = self.proj(lstm_out[:, -1])
        x = rearrange(x, "(b e) d -> b e d", b=b, e=e)
        return x
