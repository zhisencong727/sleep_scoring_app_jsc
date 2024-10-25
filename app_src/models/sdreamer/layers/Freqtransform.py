from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class STFT(torch.nn.Module):
    def __init__(
        self,
        win_length=16,
        hop_length=16,
        n_fft=256,
        window_fn=torch.hamming_window,
        normalized_stft=False,
        center=True,
        pad_mode="constant",
        onesided=True,
    ):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalized_stft = normalized_stft
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        self.window = window_fn(self.win_length).cuda()

    def forward(self, x):
        """Returns the STFT generated from the input waveforms.

        Arguments
        ---------
        x : tensor
            A batch of audio signals to transform.
        """

        # Managing multi-channel stft
        if x.ndim == 5:
            b, e, t, c, d = x.shape
        else:
            b, t, c, d = x.shape
        src_x = rearrange(x, "b ... t c d -> (b ... t c) d")

        stft = torch.stft(
            src_x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            self.center,
            self.pad_mode,
            self.normalized_stft,
            self.onesided,
            return_complex=False,
        )[:, :, :, 0]
        if x.ndim == 5:
            stft = rearrange(stft, "(b e t c) d n -> b e t n (c d)", b=b, e=e, t=t, c=c)
        else:
            stft = rearrange(stft, "(b t c) d n -> b t n (c d)", b=b, t=t, c=c)
        return stft
