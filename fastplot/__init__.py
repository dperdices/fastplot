#!/usr/bin/python3
from .plots import *
from .utils import * 

def plot(data, path=None, mode = 'line', **kw):
    args = [data, path]
    if mode == 'line_multi':
        plot_line_multi(*args, **kw)
    elif mode == 'line':
        plot_line(*args, **kw)
    elif mode == 'CDF':
        plot_CDF(*args, **kw)
    elif mode == 'CDF_multi':
        plot_CDF_multi(*args, **kw)
    elif mode == 'boxplot':
        plot_boxplot(*args, **kw)    
    elif mode == 'boxplot_multi':
        plot_boxplot_multi(*args, **kw)
    elif mode == 'timeseries':
        plot_timeseries(*args, **kw)
    elif mode == 'timeseries_multi':
        plot_timeseries_multi(*args, **kw)
    elif mode == 'timeseries_stacked':
        plot_timeseries_stacked(*args, **kw)
    elif mode == 'bars':
        plot_bars(*args, **kw)
    elif mode == 'bars_multi':
        plot_bars_multi(*args, **kw)
    elif mode == 'bars_stacked':
        plot_bars_stacked(*args, **kw)
    elif mode == 'callback':
        plot_callback(plt, **kw)

    return plt

