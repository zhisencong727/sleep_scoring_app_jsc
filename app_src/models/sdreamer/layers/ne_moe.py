from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from layers.patchEncoder import PatchEncoder, SWPatchEncoder
from layers.attention import (
    MultiHeadAttention,
    MultiHeadCrossAttention,
    MoEBlock_wNE,
    MultiHeadCrossAttention2,
)
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from layers.head import Pooler, SeqPooler, SeqPooler2


class get_cls_token(nn.Module):
    def __init__(self, inner_dim, flag, front_append=False):
        super().__init__()
        self.flag = flag
        self.front_append = front_append
        cls_mapper = {
            "seq": nn.Parameter(torch.zeros(1, 1, 1, inner_dim)),
            "epoch": nn.Parameter(torch.zeros(1, 1, inner_dim)),
        }
        self.cls_token = cls_mapper[flag]
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        if self.flag == "epoch":
            cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=x.shape[0])
        else:
            cls_tokens = repeat(
                self.cls_token, "() () n d -> b e n d", b=x.shape[0], e=x.shape[1]
            )
        if self.front_append:
            x = torch.cat([cls_tokens, x], dim=-2)
        else:
            x = torch.cat([x, cls_tokens], dim=-2)
        return x


class get_pos_emb(nn.Module):
    def __init__(self, n_patches, inner_dim, flag, dropout=0.0, cls=True):
        super().__init__()
        self.flag = flag
        n_patches = n_patches + 1 if cls else n_patches
        pos_mapper = {
            "seq": nn.Parameter(torch.zeros(1, 1, n_patches, inner_dim)),
            "epoch": nn.Parameter(torch.zeros(1, n_patches, inner_dim)),
        }
        self.pos_emb = pos_mapper[flag]
        self.pos_drop = nn.Dropout(dropout)
        trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        x = x + self.pos_emb
        x = self.pos_drop(x)
        return x


class get_mod_emb(nn.Module):
    def __init__(self, inner_dim, flag, dropout=0.0):
        super().__init__()
        self.flag = flag
        mod_mapper = {
            "seq": nn.Parameter(torch.zeros(1, 1, 1, inner_dim)),
            "epoch": nn.Parameter(torch.zeros(1, 1, inner_dim)),
        }
        self.mod_emb = mod_mapper[flag]
        self.mod_drop = nn.Dropout(dropout)
        trunc_normal_(self.mod_emb, std=0.02)

    def forward(self, x):
        x = x + self.mod_emb()
        x = self.mod_drop(x)
        return x


class MoELoader(nn.Module):
    def __init__(
        self,
        feat_type,
        patch_len,
        n_patches,
        c_in,
        inner_dim,
        dropout=0.0,
        mix_type=0,
        cls=True,
        flag="epoch",
        domain="time",
        front_append=True,
    ):
        super().__init__()
        pos, mod = False, False
        if mix_type != 1:
            pos = True
        if mix_type == 2:
            mod = True

        patch_mapper = {
            "time": (
                PatchEncoder(patch_len, c_in, inner_dim)
                if feat_type != "NE"
                else SWPatchEncoder(patch_len, patch_len, c_in, inner_dim, pad=True)
            ),
            "freq": nn.Linear(129, inner_dim),
        }
        self.get_cls = (
            get_cls_token(inner_dim, flag=flag, front_append=front_append)
            if cls
            else nn.Identity()
        )
        self.get_pos = (
            get_pos_emb(n_patches, inner_dim, flag, dropout, cls)
            if pos
            else nn.Identity()
        )

        self.patch_ecncoder = patch_mapper[domain] if cls else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"get_pos", "get_cls"}

    def forward(self, x):
        x = self.patch_ecncoder(x)

        x = self.get_cls(x)
        x = self.get_pos(x)
        x_mask = torch.ones(x.shape[0], x.shape[1]).long().to(x.device)
        return x, x_mask


class EpochMoETransformer(nn.Module):
    def __init__(
        self,
        patch_len,
        ne_patch_len,
        n_patches,
        e_layers,
        c_in,
        inner_dim,
        n_heads,
        d_head,
        dropout=0.0,
        path_drop=0.0,
        activation="glu",
        norm="layernorm",
        mult=4,
        mix_type=0,
        cls=True,
        flag="epoch",
        domain="time",
        mixffn_start_layer_index=0,
        output_attentions=False,
    ):
        super().__init__()
        self.mixffn_start_layer_index = mixffn_start_layer_index

        pos, mod = False, False
        if mix_type != 1:
            pos = True
        if mix_type == 2:
            mod = True

        self.eeg_loader = MoELoader(
            "EEG",
            patch_len,
            n_patches,
            c_in,
            inner_dim,
            dropout=dropout,
            mix_type=mix_type,
            cls=cls,
            flag=flag,
            domain=domain,
        )
        self.emg_loader = MoELoader(
            "EMG",
            patch_len,
            n_patches,
            c_in,
            inner_dim,
            dropout=dropout,
            mix_type=mix_type,
            cls=cls,
            flag=flag,
            domain=domain,
        )
        self.ne_loader = MoELoader(
            "NE",
            ne_patch_len,
            n_patches,
            c_in,
            inner_dim,
            dropout=dropout,
            mix_type=mix_type,
            cls=cls,
            flag=flag,
            domain=domain,
        )
        dpr = [x.item() for x in torch.linspace(0, path_drop, e_layers)]

        self.pool = Pooler(inner_dim)
        self.mod_emb = nn.Embedding(3, inner_dim)
        self.mod_emb.apply(init_weights)

        n_patches = n_patches + 1 if cls else n_patches
        self.transformer = nn.ModuleList(
            [
                MoEBlock_wNE(
                    n_patches,
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    path_drop=dpr[i],
                    activation=activation,
                    norm=norm,
                    mult=mult,
                    with_mixffn=(i >= self.mixffn_start_layer_index),
                )
                for i in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(inner_dim)

        self.eeg_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.emg_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.ne_proj = nn.Linear(inner_dim, inner_dim, bias=False)

        self.eeg_proj.apply(init_weights)
        self.emg_proj.apply(init_weights)
        self.ne_proj.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"get_pos", "get_cls"}

    def infer(self, eeg, emg, ne):
        eeg_embs, eeg_mask = self.eeg_loader(eeg)
        emg_embs, emg_mask = self.emg_loader(emg)
        ne_embs, ne_mask = self.ne_loader(ne)
        eeg_embs, emg_embs, ne_embs = (
            eeg_embs + self.mod_emb(torch.full_like(eeg_mask, 0)),
            emg_embs + self.mod_emb(torch.full_like(emg_mask, 1)),
            ne_embs + self.mod_emb(torch.full_like(ne_mask, 2)),
        )

        co_embeds = torch.cat([eeg_embs, emg_embs, ne_embs], dim=1)
        co_masks = torch.cat([eeg_mask, emg_mask, ne_mask], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer):
            x = blk(x, mask=co_masks, modality_type="mix")

        x = self.norm(x)

        eeg_feats, emg_feats, ne_feats = (
            x[:, : eeg_embs.shape[1]],
            x[:, eeg_embs.shape[1] : eeg_embs.shape[1] + emg_embs.shape[1]],
            x[:, eeg_embs.shape[1] + emg_embs.shape[1] :],
        )

        cls_feats = self.pool(x)

        ret = {
            "eeg_feats": eeg_feats,
            "emg_feats": emg_feats,
            "ne_feats": ne_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
        }

        return ret

    def infer_eeg(self, eeg):
        eeg_embs, eeg_mask = self.eeg_loader(eeg)
        eeg_embs = eeg_embs + self.mod_emb(torch.full_like(eeg_mask, 0))

        co_embeds = eeg_embs
        co_masks = eeg_mask

        x = co_embeds
        all_hidden_states = []
        for i, blk in enumerate(self.transformer):
            x = blk(x, mask=co_masks, modality_type="eeg")
            all_hidden_states.append(x)

        eeg_hiddens = all_hidden_states[-1]

        eeg_hiddens = self.norm(eeg_hiddens)

        eeg_feats, emg_feats, ne_feats = (
            eeg_hiddens,
            None,
            None,
        )
        cls_feats = self.eeg_proj(eeg_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "eeg_feats": eeg_feats,
            "emg_feats": emg_feats,
            "ne_feats": ne_feats,
            "cls_feats": cls_feats,
            "cls_mixffn_feats": None,
            "raw_cls_feats": eeg_hiddens[:, 0],
        }

        return ret

    def infer_emg(self, emg):
        emg_embs, emg_mask = self.emg_loader(emg)
        emg_embs = emg_embs + self.mod_emb(torch.full_like(emg_mask, 1))

        co_embeds = emg_embs
        co_masks = emg_mask

        x = co_embeds
        all_hidden_states = []
        for i, blk in enumerate(self.transformer):
            x = blk(x, mask=co_masks, modality_type="emg")
            all_hidden_states.append(x)

        emg_hiddens = all_hidden_states[-1]

        emg_hiddens = self.norm(emg_hiddens)

        eeg_feats, emg_feats, ne_feats = (
            None,
            emg_hiddens,
            None,
        )
        cls_feats = self.emg_proj(emg_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "eeg_feats": eeg_feats,
            "emg_feats": emg_feats,
            "ne_feats": ne_feats,
            "cls_feats": cls_feats,
            "cls_mixffn_feats": None,
            "raw_cls_feats": emg_hiddens[:, 0],
        }

        return ret

    def infer_ne(self, ne):
        ne_embs, ne_mask = self.ne_loader(ne)
        ne_embs = ne_embs + self.mod_emb(torch.full_like(ne_mask, 2))

        co_embeds = ne_embs
        co_masks = ne_mask

        x = co_embeds
        all_hidden_states = []
        for i, blk in enumerate(self.transformer):
            x = blk(x, mask=co_masks, modality_type="ne")
            all_hidden_states.append(x)

        ne_hiddens = all_hidden_states[-1]

        ne_hiddens = self.norm(ne_hiddens)

        eeg_feats, emg_feats, ne_feats = (
            None,
            None,
            ne_hiddens,
        )
        cls_feats = self.ne_proj(ne_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "eeg_feats": eeg_feats,
            "emg_feats": emg_feats,
            "ne_feats": ne_feats,
            "cls_feats": cls_feats,
            "cls_mixffn_feats": None,
            "raw_cls_feats": ne_hiddens[:, 0],
        }

        return ret


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
