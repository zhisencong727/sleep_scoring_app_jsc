# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 2024

@author: zcong and yzhao
"""

import glob
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from app_src.models.sdreamer import n2nBaseLineNE
from app_src.preprocessing_ne import reshape_sleep_data_ne


class SequenceDataset(Dataset):
    def __init__(self, normalized_sleep_data, normalized_ne_data):
        self.traces = normalized_sleep_data
        self.nes = normalized_ne_data

    def __len__(self):
        return self.traces.shape[0]

    def __getitem__(self, idx):
        trace = self.traces[idx]
        ne = self.nes[idx]
        return trace,ne


def make_dataset(data: dict, n_sequences: int = 64):
    eeg, emg, ne = reshape_sleep_data_ne(data)

    sleep_data = np.stack((eeg, emg), axis=1)
    sleep_data = torch.from_numpy(sleep_data)
    sleep_data = torch.unsqueeze(sleep_data, dim=2)  # shape [n_seconds, 2, 1, seq_len]
    mean, std = torch.mean(sleep_data, dim=0), torch.std(sleep_data, dim=0)
    normalized_data = (sleep_data - mean) / std

    n_seconds = normalized_data.shape[0]
    n_to_crop = n_seconds % n_sequences
    if n_to_crop != 0:
        normalized_data = torch.cat(
            [normalized_data[:-n_to_crop], normalized_data[-n_sequences:]], dim=0
        )

    normalized_data = normalized_data.reshape(
        (
            -1,
            n_sequences,
            normalized_data.shape[1],
            normalized_data.shape[2],
            normalized_data.shape[3],
        )
    )
    #print("normalized_data.shape:",normalized_data.shape)

    ne_data = torch.from_numpy(ne)
    ne_mean, ne_std = torch.mean(ne_data,dim=0),torch.std(ne_data,dim=0)
    normalized_ne_data = (ne_data - ne_mean) / ne_std

    n_seconds_ne = normalized_ne_data.shape[0]
    n_to_crop_ne = n_seconds_ne % n_sequences
    #print("n_to_crop_ne:",n_to_crop_ne)
    if n_to_crop_ne != 0:
        normalized_ne_data = torch.cat(
            [normalized_ne_data[:-n_to_crop_ne], normalized_ne_data[-n_sequences:]], dim=0
        )
    #print("normalized_ne_data.shape:",normalized_ne_data.shape)
    normalized_ne_data = normalized_ne_data.reshape(
        (
            -1,
            n_sequences,
            1,
            normalized_ne_data.shape[1],
        )
    )
    #print("normalized_ne_data.shape:",normalized_ne_data.shape)

    dataset = SequenceDataset(normalized_data,normalized_ne_data)
    return dataset, n_seconds, n_to_crop


# %%

# hyperparameters
activation = "glu"
norm_type = "layernorm"
patch_len = 16
seed = 42
ca_layers = 1
batch_size = 64
n_sequences = 64
ne_patch_len = 10
e_layers = 2
fold = 1

# BaseLine_Seq_pl16_el2_cl1_f1_seql3_kl_2.0_t3.5

config = dict(
    seed=42,
    is_training=1,
    model_id="test",
    model="BaseLine",
    data="Seq",
    isNE=True,
    fold=1,
    root_path="",
    # data_path=data_path,
    features="ALL",
    n_sequences=n_sequences,
    useNorm=True,
    num_workers=10,
    seq_len=512,
    patch_len=patch_len,
    ne_patch_len=ne_patch_len,
    stride=8,
    padding_patch="end",
    subtract_last=0,
    decomposition=0,
    kernel_size=25,
    individual=0,
    mix_type=0,
    c_out=3,
    d_model=128,
    n_heads=8,
    e_layers=e_layers,
    ca_layers=ca_layers,
    seq_layers=3,
    d_ff=512,
    dropout=0.1,
    path_drop=0.0,
    pos_emb="learned",
    activation=activation,
    norm_type=norm_type,
    output_attentions=False,
    useRaw=False,
    epochs=100,
    batch_size=batch_size,
    patience=30,
    optimizer="adamw",
    lr=0.001,
    weight_decay=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    eps=1e-9,
    scheduler="CosineLR",
    scale=0.0,
    pct_start=0.3,
    step_size=30,
    gamma=0.5,
    weight=[1, 1, 1],
    visualize_mode=[],
    visualizations="",
    # checkpoints=checkpoints,
    reload_best=True,
    reload_ckpt=None,
    use_gpu=True,
    gpu=0,
    use_multi_gpu=False,
    test_flop=False,
    print_freq=50,
    # output_path=output_path,
    # ne_patch_len=ne_patch_len,
    # des=des_name,
    # pad=False,
)

def build_args(**kwargs):
    parser = argparse.ArgumentParser(description="Transformer family for sleep scoring")
    args = parser.parse_args()
    parser_dict = vars(args)

    for k, v in config.items():
        parser_dict[k] = v
    for k, v in kwargs.items():
        parser_dict[k] = v
    return args

# %%
def infer(data, model_path, batch_size=64):
    args = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = n2nBaseLineNE.Model(args)
    model = model.to(device)
    checkpoint_path = glob.glob(model_path + "*.tar")[0]
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    n_sequences = config["n_sequences"]
    dataset, n_seconds, n_to_crop = make_dataset(data)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    model.eval()
    with torch.no_grad():
        all_pred = []
        all_prob = []

        with tqdm(total=n_seconds, unit=" seconds of signal") as pbar:
            for batch, (traces,nes) in enumerate(data_loader, 1):
                traces = traces.to(device)  # [batch_size, 64, 2, 1, 512]
                nes = nes.to(device)
                #print("nes.shape",nes.shape)
                #nes = torch.zeros(nes.shape[0],64,1,10)
                out_dict = model(traces, nes, label=None)
                out = out_dict["out"]

                prob = torch.max(torch.softmax(out, dim=1), dim=1).values
                all_prob.append(prob.detach().cpu())

                pred = np.argmax(out.detach().cpu(), axis=1)
                # pred = out_dict["predictions"]
                all_pred.append(pred)

                pbar.update(batch_size * n_sequences)
            pbar.set_postfix({"Number of batches": batch})

        if n_to_crop != 0:
            all_pred[-1] = torch.cat(
                (
                    all_pred[-1][: -args.n_sequences],
                    all_pred[-1][-args.n_sequences :][-n_to_crop:],
                )
            )
            all_prob[-1] = torch.cat(
                (
                    all_prob[-1][: -args.n_sequences],
                    all_prob[-1][-args.n_sequences :][-n_to_crop:],
                )
            )

        all_pred = np.concatenate(all_pred)
        all_prob = np.concatenate(all_prob)

    return all_pred, all_prob


if __name__ == "__main__":
    from scipy.io import loadmat

    model_path = "../models/sdreamer/checkpoints/"
    mat_file = "../user_test_files/sal_588.mat"
    data = loadmat(mat_file)
    all_pred, all_prob = infer(data, model_path)