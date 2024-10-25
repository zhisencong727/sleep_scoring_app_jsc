# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:44:51 2024

@author: yzhao
"""

from threading import Timer
from functools import partial

from app import app, open_browser


if __name__ == "__main__":
    PORT = 8050
    VERSION = "v0.10.1"
    Timer(1, partial(open_browser, PORT)).start()
    app.title = "Sleep Scoring App v0.11.0"
    app.run_server(debug=False, port=PORT)
