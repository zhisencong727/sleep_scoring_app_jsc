import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class LinearPatchEncoder(nn.Module):
    # Batch x Epoch x Trace x Channel x Time
    # Batch x Epoch x Trace x Channel x (Patch_num x Patch_len)
    # Batch x Epoch x Trace x Patch_num x (Channel x Patch_len)

    def __init__(self, trace_id=0, patch_len=16, in_channel=1, d_model=128):
        super().__init__()

        self.trace_id = trace_id
        self.patch_dim = patch_len * in_channel
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b ... c (n l) -> b ... n (l c)", l=patch_len),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        trace = x[:, :, self.trace_id] if x.ndim == 5 else x[:, self.trace_id]
        trace_emb = self.to_patch_embedding(trace)
        return trace_emb


class LinearPatchEncoder2(nn.Module):
    # Batch x Epoch x Trace x Channel x Time
    # Batch x Epoch x Trace x Channel x (Patch_num x Patch_len)
    # Batch x Epoch x Trace x Patch_num x (Channel x Patch_len)

    def __init__(self, trace_id=0, patch_len=16, in_channel=1, d_model=128):
        super().__init__()

        self.trace_id = trace_id
        self.patch_dim = patch_len * in_channel
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b ... c (n l) -> b ... n (l c)", l=patch_len),
            nn.Linear(self.patch_dim, d_model, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, x):
        trace = x[:, :, self.trace_id] if x.ndim == 5 else x[:, self.trace_id]
        trace_emb = self.to_patch_embedding(trace)  # 128 * 32 * 128
        return trace_emb


class CNNPatchEncoder(nn.Module):
    # Batch x Epoch x Trace x Channel x Time
    # Batch x Epoch x Trace x Channel x (Patch_num x Patch_len)
    # Batch x Epoch x Trace x Patch_num x (Channel x Patch_len)

    def __init__(self, trace_id=0):
        super().__init__()

        self.trace_id = trace_id
        # 128 * 1 * 512
        self.encoder_1 = nn.Sequential(
            nn.Conv1d(1, 8, 8, 4, 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv1d(8, 64, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
        )
        self.linear = nn.Linear(128, 128)

    def forward(self, x):
        # 128 * 1 * 512
        trace = x[:, :, self.trace_id] if x.ndim == 5 else x[:, self.trace_id]
        trace_emb = self.encoder_1(trace)
        trace_emb = self.encoder_2(trace_emb)
        trace_emb = self.encoder_3(trace_emb)
        trace_emb = torch.transpose(trace_emb, -1, -2)
        trace_emb = self.linear(trace_emb)
        return trace_emb


class PatchEncoder(nn.Module):
    # Batch x Epoch x Trace x Channel x Time
    # Batch x Epoch x Trace x Channel x (Patch_num x Patch_len)
    # Batch x Epoch x Trace x Patch_num x (Channel x Patch_len)

    def __init__(self, patch_len=16, in_channel=1, d_model=128):
        super().__init__()

        self.patch_dim = patch_len * in_channel
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b ... c (n l) -> b ... n (l c)", l=patch_len),
            nn.Linear(self.patch_dim, d_model, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, x):
        #print("x.shape ERROR HERE is:",x.shape)
        x = self.to_patch_embedding(x)  # 128 * 32 * 128
        return x


class SWPatchEncoder(nn.Module):
    # Batch x Epoch x Trace x Channel x Time
    # Batch x Epoch x Trace x Channel x (Patch_num x Patch_len)
    # Batch x Epoch x Trace x Patch_num x (Channel x Patch_len)

    def __init__(self, patch_len=16, stride=8, in_channel=1, d_model=128, pad=True):
        super().__init__()

        self.patch_dim = patch_len * in_channel
        #print("self.patch_dim here is:",self.patch_dim)
        self.stride = stride

        # pad 0 for stride length
        self.padd_layer = nn.ConstantPad1d((0, stride),0) if pad else nn.Identity()

        self.patch_spliter = nn.Sequential(
            Rearrange("b ... c t -> b ... (c t)"),
            self.padd_layer,
        )

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.patch_dim, d_model, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, x):
        print("x.shape at the beginning of patchEncoder is: ",x.shape)
        x = self.patch_spliter(x)
        print("x.shape after patch_splitter is: ",x.shape)
        #print(x)
        x = x.unfold(dimension=-1, size=self.patch_dim, step=self.stride)
        print("x.shape after unfold is: ",x.shape)
        x = self.to_patch_embedding(x)  
        print("x.shape after to_patch_embedding is: ",x.shape)
        return x
