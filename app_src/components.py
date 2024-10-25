# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:27:03 2023

@author: yzhao
"""

from dash import dcc, html

# from trace_updater import TraceUpdater
from dash_extensions import EventListener


home_div = html.Div(
    [
        html.Div(
            style={"display": "flex"},
            children=[
                html.Div(
                    dcc.RadioItems(
                        id="task-selection",
                        options=[
                            {"label": "Generate prediction", "value": "gen"},
                            {"label": "Visualize existing prediction", "value": "vis"},
                        ],
                        style={"marginRight": "50px"},
                    )
                ),
                html.Div(
                    id="model-choice-container",
                    children=[
                        dcc.RadioItems(
                            id="model-choice",
                            options=[
                                # {"label": "MSDA by Shadi", "value": "msda"},
                                {"label": "sDREAMER", "value": "sdreamer"},
                            ],
                            value="sdreamer",
                        )
                    ],
                    style={"display": "none"},
                ),
                html.Div([" "], id="invisible-gap", style={"marginRight": "50px"}),
                html.Div(
                    id="num-class-container",
                    children=[
                        dcc.RadioItems(
                            id="num-class-choice",
                            options=[
                                {"label": "3 Classes", "value": 3},
                                # {"label": "4 Classes", "value": 4},
                            ],
                            value=3,
                        )
                    ],
                    style={"display": "none"},
                ),
            ],
        ),
        html.Div(id="upload-container"),
        html.Div(id="data-upload-message"),
        dcc.Store(id="model-choice-store"),
        dcc.Store(id="num-class-store"),
        dcc.Store(id="extension-validation-store"),
        dcc.Store(id="generation-ready-store"),
        dcc.Store(id="visualization-ready-store"),
        dcc.Download(id="prediction-download-store"),
    ]
)

mat_upload_box = dcc.Upload(
    id="data-upload",
    children=html.Button(
        ["Click here to select File"],
        id="select-file-button",
    ),
    style={
        "width": "12%",
        "height": "auto",
        "borderWidth": "0px",
        "borderStyle": "dashed",
        "textAlign": "left",
        "margin": "5px",
    },
    multiple=False,
)

graph = dcc.Graph(
    id="graph",
    config={
        "scrollZoom": True,
        "editable": True,
        "edits": {
            "axisTitleText": False,
            "titleText": False,
            "colorbarTitleText": False,
            "annotationText": False,
        },
    },
)


visualization_div = html.Div(
    children=[
        html.Div(
            style={"display": "flex"},
            children=[
                html.Div(
                    ["Sampling Level"],
                    style={
                        "marginRight": "10px",
                        "lineHeight": "40px",
                    },
                ),
                html.Div(
                    children=[
                        dcc.Dropdown(
                            ["x1", "x2", "x4"],
                            "x1",
                            id="n-sample-dropdown",
                            # placeholder="Select a sampling level",
                        )
                    ],
                    style={
                        "width": "50px",
                        "height": "auto",
                        "textAlign": "left",
                        "margin": "2px",
                        "marginRight": "20px",
                    },
                ),
            ],
        ),
        graph,
        # TraceUpdater(id="trace-updater", gdID="graph"),
        html.Div(
            style={"display": "flex"},
            children=[
                dcc.Store(id="box-select-store"),
                dcc.Store(id="annotation-store"),
                EventListener(
                    id="keyboard",
                    events=[{"event": "keydown", "props": ["key"]}],
                ),
                html.Div(
                    style={"display": "flex", "margin-right": "5px"},
                    children=[
                        html.Button("Save Annotations", id="save-button"),
                        dcc.Download(id="download-annotations"),
                        dcc.Download(id="download-spreadsheet"),
                        html.Button(
                            "Undo Annotation",
                            id="undo-button",
                            style={"display": "none"},
                        ),
                    ],
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=1 * 1000,  # in milliseconds
                    max_intervals=0,  # stop after the first interval
                ),
                html.Div(id="annotation-message"),
                html.Div(id="debug-message"),
            ],
        ),
    ],
)


# %%
class Components:
    def __init__(self):
        self.home_div = home_div
        self.mat_upload_box = mat_upload_box
        self.graph = graph
        self.visualization_div = visualization_div
