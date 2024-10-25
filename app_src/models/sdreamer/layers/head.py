import torch
from torch import nn

from app_src.models.sdreamer.layers import transformer


class Pooler(nn.Module):
    def __init__(self, hidden_size, useRaw=False):
        super().__init__()
        self.pool_head = (
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
            if not useRaw
            else nn.Identity()
        )
        self.pool_head.apply(transformer.init_weights)

    def forward(self, hidden_states):
        last_token_tensor = (
            hidden_states[:, 0] if hidden_states.ndim == 3 else hidden_states[:, :, 0]
        )
        # print(first_token_tensor.shape)
        pooled_output = self.pool_head(last_token_tensor)
        return pooled_output


class SeqPooler(nn.Module):
    def __init__(self, hidden_size, useRaw=False):
        super().__init__()
        self.pool_head = (
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
            if not useRaw
            else nn.Identity()
        )
        self.pool_head.apply(transformer.init_weights)

    def forward(self, hidden_states):
        pooled_output = self.pool_head(hidden_states)
        return pooled_output


class cls_head(nn.Module):
    def __init__(self, hidden_size, c_out=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, c_out),
        )
        self.mlp.apply(transformer.init_weights)

    def forward(self, x):
        x = self.mlp(x)
        return x


class SeqPooler2(nn.Module):
    def __init__(self, hidden_size, useRaw=False):
        super().__init__()
        self.pool_head = (
            nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh())
            if not useRaw
            else nn.Identity()
        )
        self.pool_head.apply(transformer.init_weights)

    def forward(self, hidden_states):
        hidden_states = torch.cat(
            (
                hidden_states[:, : hidden_states.shape[1] // 2],
                hidden_states[:, hidden_states.shape[1] // 2 :],
            ),
            dim=-1,
        )
        pooled_output = self.pool_head(hidden_states)
        return pooled_output
