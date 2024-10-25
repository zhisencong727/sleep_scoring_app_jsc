# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:36:14 2023

@author: Yue

Notes
1. A common reason that sleep scores and confidence, both of which are heatmaps,
   don't show up is that they have shape of (N,), instead of (1, N). The heatmap
   only works with 2d arrays.
"""

import math
import numpy as np

# from scipy import signal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB


# set up color config
sleep_score_opacity = 1
stage_colors = [
    "rgb(124, 124, 251)",  # Wake,
    "rgb(251, 124, 124)",  # SWS,
    "rgb(123, 251, 123)",  # REM,
    "rgb(255, 255, 0)",  # MA yellow
]
stage_names = ["Wake: 1", "SWS: 2", "REM: 3", "MA: 4"]
colorscale = {
    3: [[0, stage_colors[0]], [0.5, stage_colors[1]], [1, stage_colors[2]]],
    4: [
        [0, stage_colors[0]],
        [1 / 3, stage_colors[1]],
        [2 / 3, stage_colors[2]],
        [1, stage_colors[3]],
    ],
}
range_quantile = 0.9999
range_padding_percent = 0.2


def make_figure(mat, mat_name="", default_n_shown_samples=4000, ne_fs=10):
    # Time span and frequencies
    eeg, emg, ne = mat.get("eeg"), mat.get("emg"), mat.get("ne")
    eeg, emg = eeg.flatten(), emg.flatten()
    eeg_freq, ne_freq = mat.get("eeg_frequency"), mat.get("ne_frequency")
    eeg_freq = eeg_freq.item()
    start_time = 0
    eeg_end_time = (eeg.size - 1) / eeg_freq
    # Create the time sequences
    time_eeg = np.linspace(start_time, eeg_end_time, eeg.size)
    eeg_end_time = math.ceil(eeg_end_time)
    time = np.expand_dims(np.arange(1, eeg_end_time + 1), 0)
    eeg_lower_range, eeg_upper_range = np.quantile(
        eeg, 1 - range_quantile
    ), np.quantile(eeg, range_quantile)
    emg_lower_range, emg_upper_range = np.quantile(
        emg, 1 - range_quantile
    ), np.quantile(emg, range_quantile)
    eeg_range = max(abs(eeg_lower_range), abs(eeg_upper_range))
    emg_range = max(abs(emg_lower_range), abs(emg_upper_range))

    labels = mat.get("pred_labels")
    if labels is None or labels.size == 0:
        # either scored manually or unscored
        labels = mat.get("sleep_scores")
        if labels is None or labels.size == 0:
            # if unscored, initialize with nan, set confidence to be zero
            mat["sleep_scores"] = np.zeros((1, eeg_end_time))
            mat["sleep_scores"][:] = np.nan
            mat["confidence"] = np.zeros((1, eeg_end_time))
            labels = mat["sleep_scores"]
        else:  # manually scored, but may contain missing scores
            # make a labels copy and do not modify mat. only need to replace
            # -1 in labels copy with nan for visualization

            # sleep_scores will have the length of eeg_end_time. this is
            # guaranteed in the preprocessing process.
            labels = labels.copy()
            labels = labels.astype(float)
            np.place(
                labels, labels == -1, [np.nan]
            )  # convert -1 to None for heatmap visualization
            mat["confidence"] = np.ones((1, labels.size))
            mat["confidence"][np.isnan(labels)] = 0.0

    # if pred_labels exists, then there is confidence
    confidence = mat.get("confidence")

    # convert flat array to 2D array for visualization to work
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=0)
    if len(confidence.shape) == 1:
        confidence = np.expand_dims(confidence, axis=0)
    num_class = mat["num_class"].item()

    fig = FigureResampler(
        make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "EEG",
                "EMG",
                "NE",
                "Prediction Confidence",
            ),
            row_heights=[0.3, 0.3, 0.3, 0.1],
        ),
        default_n_shown_samples=default_n_shown_samples,
        default_downsampler=MinMaxLTTB(parallel=True),
    )

    ne_lower_range, ne_upper_range = 0, 0
    if ne.size > 1:
        ne = ne.flatten()
        ne_freq = ne_freq.item()
        ne_end_time = (ne.size - 1) / ne_freq

        # Create the time sequences
        time_ne = np.linspace(start_time, ne_end_time, ne.size)
        ne_end_time = math.ceil(ne_end_time)
        ne_lower_range, ne_upper_range = np.quantile(
            ne, 1 - range_quantile
        ), np.quantile(ne, range_quantile)
        fig.add_trace(
            go.Scattergl(
                line=dict(width=1),
                marker=dict(size=2, color="black"),
                showlegend=False,
                mode="lines+markers",
                hovertemplate="<b>time</b>: %{x:.2f}"
                + "<br><b>y</b>: %{y}<extra></extra>",
            ),
            hf_x=time_ne,
            hf_y=ne,
            row=3,
            col=1,
        )

    ne_range = max(abs(ne_lower_range), abs(ne_upper_range))
    heatmap_width = max(
        20, 2 * (1 + range_padding_percent) * max([eeg_range, emg_range, ne_range])
    )

    # Create a heatmap for stages
    sleep_scores = go.Heatmap(
        x0=0.5,
        dx=1,
        y0=0,
        dy=heatmap_width,  # assuming that the max abs value of eeg, emg, or ne is no more than 10
        z=labels,
        hoverinfo="none",
        colorscale=colorscale[num_class],
        showscale=False,
        opacity=sleep_score_opacity,
        zmax=num_class - 1,
        zmin=0,
        showlegend=False,
        xgap=0.05,  # add small gaps to serve as boundaries / ticks
    )

    conf = go.Heatmap(
        x0=0.5,
        dx=1,
        z=confidence,
        customdata=time,
        hovertemplate="<b>time</b>: %{customdata}<extra></extra>",
        colorscale="speed",
        zmax=1,
        zmin=0,
        colorbar=dict(
            thicknessmode="fraction",  # set the mode of thickness to fraction
            thickness=0.005,  # the thickness of the colorbar
            lenmode="fraction",  # set the mode of length to fraction
            len=0.15,  # the length of the colorbar
            yanchor="bottom",  # anchor the colorbar at the top
            y=0.08,  # the y position of the colorbar
            xanchor="right",  # anchor the colorbar at the left
            x=0.75,  # the x position of the colorbar
            tickfont=dict(size=8),
        ),
        showscale=True,
        xgap=0.05,  # add small gaps to serve as boundaries / ticks
    )

    # Add the time series to the figure
    fig.add_trace(
        go.Scattergl(
            line=dict(width=1),
            marker=dict(size=2, color="black"),
            showlegend=False,
            mode="lines+markers",
            hovertemplate="<b>time</b>: %{x:.2f}" + "<br><b>y</b>: %{y}<extra></extra>",
        ),
        hf_x=time_eeg,
        hf_y=eeg,
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            line=dict(width=1),
            marker=dict(size=2, color="black"),
            showlegend=False,
            mode="lines+markers",
            hovertemplate="<b>time</b>: %{x:.2f}" + "<br><b>y</b>: %{y}<extra></extra>",
        ),
        hf_x=time_eeg,
        hf_y=emg,
        row=2,
        col=1,
    )

    for i, color in enumerate(stage_colors[:num_class]):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[0.2],
                mode="markers",
                marker=dict(
                    size=8, color=color, symbol="square", opacity=sleep_score_opacity
                ),
                name=stage_names[i],
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # add the heatmap last so that their indices can be accessed using last indices
    fig.add_trace(sleep_scores, row=1, col=1)
    fig.add_trace(sleep_scores, row=2, col=1)
    fig.add_trace(sleep_scores, row=3, col=1)
    fig.add_trace(conf, row=4, col=1)

    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=20, r=20, b=40),
        height=800,
        hovermode="x unified",  # gives crosshair in one subplot
        hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.6)"),
        title_text=mat_name,
        yaxis4=dict(tickvals=[]),  # suppress y ticks on the heatmap
        xaxis4=dict(tickformat="digits"),
        legend=dict(
            x=0.6,  # adjust these values to position the sleep score legend stage_names
            y=1.05,
            orientation="h",  # makes legend items horizontal
            bgcolor="rgba(0,0,0,0)",  # transparent legend background
            font=dict(size=10),  # adjust legend text size
        ),
        font=dict(
            size=12,  # title font size
        ),
        modebar_remove=["lasso2d", "zoom", "autoScale"],
        dragmode="pan",
        clickmode="event",
    )

    fig.update_traces(xaxis="x4")  # gives crosshair across all subplots
    fig.update_traces(colorbar_orientation="h", selector=dict(type="heatmap"))
    fig.update_xaxes(range=[start_time, eeg_end_time], row=1, col=1)
    fig.update_xaxes(range=[start_time, eeg_end_time], row=2, col=1)
    fig.update_xaxes(range=[start_time, eeg_end_time], row=3, col=1)
    fig.update_xaxes(
        range=[start_time, eeg_end_time],
        row=4,
        col=1,
        title_text="<b>Time (s)</b>",
    )
    fig.update_yaxes(
        range=[
            eeg_range * -(1 + range_padding_percent),
            eeg_range * (1 + range_padding_percent),
        ],
        # fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[
            emg_range * -(1 + range_padding_percent),
            emg_range * (1 + range_padding_percent),
        ],
        # fixedrange=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[
            ne_range * -(1 + range_padding_percent),
            ne_range * (1 + range_padding_percent),
        ],
        # fixedrange=True,
        row=3,
        col=1,
    )
    fig.update_yaxes(range=[0, 0.5], fixedrange=True, row=4, col=1)
    fig.update_annotations(font_size=14)  # subplot title size
    fig["layout"]["annotations"][-1]["font"]["size"] = 14

    return fig


if __name__ == "__main__":
    import os
    import plotly.io as io
    from scipy.io import loadmat

    io.renderers.default = "browser"
    data_path = ".\\610Hz data\\"
    mat_file = "20240808_3_FP_Temp_BS_rep.mat"
    mat = loadmat(os.path.join(data_path, mat_file))
    # mat_file = "C:/Users/yzhao/matlab_projects/sleep_data_extraction/2023-10-17_Day1_no_stim_705/2023-10-17_Day1_no_stim_705.mat"

    # mat = loadmat(mat_file)
    mat_name = os.path.basename(mat_file)
    fig = make_figure(mat, mat_name=mat_name)
    fig.show_dash(config={"scrollZoom": True})
