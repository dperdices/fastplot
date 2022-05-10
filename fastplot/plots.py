from threading import local
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cycler import cycler
import seaborn as sns
import numpy as np
import pandas as pd
import re
from statsmodels.distributions.empirical_distribution import ECDF

# Register Pandas Converters
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from .utils import *

FIGSIZE=(4,2.25)
NUM_BIN_CDF=1000
CYCLER_LINES=(cycler('color', ['r', 'b', 'g', 'purple', 'c']) +
              cycler('linestyle', ['-', '--', '-.', ':', (0, (3, 1, 1, 1)) ]))
CYCLER_LINESPOINTS=(cycler('color', ['r', 'b', 'g', 'purple', 'c']) +
                    cycler('linestyle', ['-', '--', '-.', ':', (0, (3, 1, 1, 1)) ]) +
                    cycler('marker', ['o', 's', 'v', 'd', '^' ]))
CYCLER_POINTS=(cycler('color', ['r', 'b', 'g', 'purple', 'c']) +
               cycler('linestyle', ['', '', '', '', '']) +
               cycler('marker', ['o', 's', 'v', 'd', '^' ]))

CYCLER_LINES_BLACK=(cycler('color', ['black', 'black', 'black', 'black', 'black']) +
                    cycler('linestyle', ['-', '--', '-.', ':', (0, (3, 1, 1, 1)) ]))
CYCLER_LINESPOINTS_BLACK=(cycler('color', ['black', 'black', 'black', 'black', 'black']) +
                    cycler('linestyle', ['-', '--', '-.', ':', (0, (3, 1, 1, 1)) ]) +
                    cycler('marker', ['o', 's', 'v', 'd', '^' ]))
CYCLER_POINTS_BLACK=(cycler('color', ['black', 'black', 'black', 'black', 'black']) +
               cycler('linestyle', ['', '', '', '', '']) +
               cycler('marker', ['o', 's', 'v', 'd', '^' ]))


def configuration_preplot(data=None, path=None, style = 'sans-serif', figsize = FIGSIZE, cycler = CYCLER_LINES, fontsize = 11, dpi=300, classic_autolimit=True, rcParams={},
         grid = False, grid_which='major', grid_axis = 'both', grid_linestyle = 'dotted', grid_color = 'black',
         yscale = 'linear' , xscale = 'linear', **kw):
    """
    Configure the environment before the plot.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.clf()
    plt.figure(figsize=figsize)

    plt.rc('axes', prop_cycle=cycler)
    plt.rc('font', **{'size': fontsize})

    # Old default axis lim
    if classic_autolimit:
        plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0

    if style == 'latex':
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        #plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

        plt.rc('text', usetex=True)
    elif style == 'serif':
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "sans-serif"

    # Set custom rcparams
    plt.rcParams.update(rcParams)


    # 2. Set axis characteristics
    plt.yscale(yscale)
    plt.xscale(xscale)
    if grid:
        plt.grid(which=grid_which, axis=grid_axis, linestyle=grid_linestyle, color=grid_color)


def configuration_postplot(data=None, path=None, mode = 'line', plt=plt,
         xlim = None, ylim = None, xlabel = None, ylabel = None, xticks = None, yticks = None, xticks_rotate = None, yticks_rotate = None, xticks_fontsize='medium', yticks_fontsize='medium', 
         xtick_direction = 'in', xtick_width = 1, xtick_length = 3, ytick_direction = 'in', ytick_width = 1, ytick_length = 3, 
         legend = False, legend_loc = 'best', legend_ncol = 1, legend_fontsize = 'medium', legend_border = False, legend_frameon = True, legend_fancybox = False, legend_alpha=1.0, legend_args = {},
        timeseries_stacked_right_legend_order=True, dpi=300, **kw):
    """
    Configure the environment after the plot.
    """

    # 4. Set axis
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)

    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])
    if yticks is not None:
        plt.yticks(yticks[0], yticks[1])
    if xticks_rotate is not None:
        if xticks_rotate > 0:
            plt.xticks(rotation=xticks_rotate, ha="right")
        else:
            plt.xticks(rotation=xticks_rotate, ha="left")
    if yticks_rotate is not None:
        if yticks_rotate > 0:
            plt.yticks(rotation=yticks_rotate, ha="right")
        else:
            plt.yticks(rotation=yticks_rotate, ha="left")

    # Tick marker params
    plt.tick_params(axis = 'x', direction = xtick_direction, width = xtick_width, length = xtick_length )
    plt.tick_params(axis = 'y', direction = ytick_direction, width = ytick_width, length = ytick_length )

    # 5. Legend
    if legend:
        legend = plt.legend(loc=legend_loc, ncol = legend_ncol, fontsize = legend_fontsize,
                            numpoints=1, frameon = legend_frameon, fancybox=legend_fancybox,
                            **legend_args)
        legend.get_frame().set_alpha(legend_alpha)
        if legend_border == False:
            legend.get_frame().set_linewidth(0.0)

        # Handle timeseries_stacked_right_legend_order
        if mode == 'timeseries_stacked' and timeseries_stacked_right_legend_order: 
            handles, labels = plt.gca().get_legend_handles_labels()
            legend = plt.gca().legend(handles[::-1], labels[::-1], loc=legend_loc, ncol = legend_ncol,
                                fontsize = legend_fontsize, numpoints=1, frameon = legend_frameon,
                                fancybox=legend_fancybox, **legend_args) 
            legend.get_frame().set_alpha(legend_alpha)
            if legend_border == False:
                legend.get_frame().set_linewidth(0.0)


    # 6. Save Fig
    plt.tight_layout()

    # Handle Interactive Plot
    if path is not None:
        plt.savefig(path, dpi=dpi)
        plt.close()
    else:
        return plt

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_line_multi(data, path=None, linewidth=1, plot_args={}, **kw):
    """
    Plot multiple lines.

    data: list of tuples (x, y).
    path (str, optional): path to save the figure.
    linewidth (int, optional): line width.
    plot_args (dict, optional): dict of plot arguments.
    """

    for name, points in data:
        plt.plot(points[0], points[1], label = name, markeredgewidth=0,
                linewidth = linewidth, **plot_args)

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_line(data, path=None, linewidth=1, plot_args={}, **kw):
    """
    Plot a line.

    data: tuple (x, y).
    path (str, optional): path to save the figure.
    linewidth (int, optional): line width.
    plot_args (dict, optional): dict of plot arguments.
    """

    plt.plot(data[0], data[1], markeredgewidth=0, linewidth = linewidth, **plot_args) 

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_CDF(data, path=None, xscale="linear", CDF_complementary=False, linewidth=1, plot_args={}, ylabel=None, ylim=(0,1), **kw):
    """
    Plot a CDF.

    data: array of values.
    path (str, optional): path to save the figure.
    xscale: scale of the x axis (linear, log).
    CDF_complementary: if True, plot the complementary CDF.
    linewidth (int, optional): line width.
    plot_args (dict, optional): dict of plot arguments.
    ylabel: label of the y axis.
    ylim: limits of the y axis.
    """

    s = data
    e = ECDF(s)
    if xscale == 'log':
        x = np.logspace(np.log10(min(s)), np.log10(max(s)), NUM_BIN_CDF )
        if CDF_complementary:
            y = 1-e(x)
        else:
            y = e(x)
    else:
        x = np.linspace(min(s), max(s), NUM_BIN_CDF )  
        if CDF_complementary:
            y = 1-e(x)
            x = np.concatenate( (np.array([min(s)]), x) )
            y = np.concatenate( (np.array([1]), y) )
        else:
            y = e(x)
            x = np.concatenate( (np.array([min(s)]), x) )
            y = np.concatenate( (np.array([0]), y) )

    plt.plot(x,y, linewidth = linewidth, **plot_args)
    if ylabel is None:
        ylabel = 'CCDF' if CDF_complementary else "CDF"
    if ylim is None:
        ylim = (0,1)

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_CDF_multi(data, path=None, xscale="linear", CDF_complementary=False, linewidth=1, plot_args={}, ylabel=None, ylim=(0,1), **kw):
    """
    Plot multiple CDFs.
    
    data: list of tuples (name, series) where series is an array.
    path (str, optional): path to save the figure.
    xscale: scale of the x axis (linear, log).
    CDF_complementary: if True, plot the complementary CDF.
    linewidth (int, optional): line width.
    plot_args (dict, optional): dict of plot arguments.
    ylabel: label of the y axis.
    ylim: limits of the y axis.
    """

    for s_name, s in data :
        e = ECDF(s)
        if xscale == 'log':
            x = np.logspace(np.log10(min(s)), np.log10(max(s)), NUM_BIN_CDF )
            if CDF_complementary:
                y = 1-e(x)
            else:
                y = e(x)
        else:
            x = np.linspace(min(s), max(s), NUM_BIN_CDF )  

            if CDF_complementary:
                y = 1-e(x)
                x = np.concatenate( (np.array([min(s)]), x) )
                y = np.concatenate( (np.array([1]), y) )
            else:
                y = e(x)
                x = np.concatenate( (np.array([min(s)]), x) )
                y = np.concatenate( (np.array([0]), y) )

        plt.plot(x,y, label=s_name, linewidth = linewidth, **plot_args)

    if ylabel is None:
        ylabel = 'CCDF' if CDF_complementary else "CDF"
    if ylim is None:
        ylim = (0,1)

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_boxplot(data, path=None, plot_args = {},
         boxplot_sym='', boxplot_whis=[5,95], 
         boxplot_numerousness = False, boxplot_numerousness_fontsize = 'x-small',
         boxplot_palette=sns.color_palette(), boxplot_empty=False, boxplot_numerousness_rotate=None, **kw):
    """
    Plot a boxplot.
    
    data: list of tuples (name, series) where series is an array.
    path (str, optional): path to save the figure.
    plot_args (dict, optional): dict of plot arguments.
    boxplot_sym: symbol of the boxplot.
    boxplot_whis (list[2]<int>, optional): whiskers.
    boxplot_numerousness (boolean, optional):  if True, plot the numerousness of the data, i.e. the number of samples per boxplot.
    boxplot_numerousness_fontsize (int, optional): font size of the numerousness.
    boxplot_palette (list of color, optional): palette of the boxplot.
    boxplot_empty (boolean, optional): if True, plot empty boxplots.
    boxplot_numerousness_rotate (int, optional): rotation of the numerousness labels.
    """

    labels = [e[0] for e in data]
    samples = [e[1] for e in data]
    #plt.boxplot(samples, labels=labels, sym=boxplot_sym, whis=boxplot_whis, **plot_args)
    
    #order = sorted(scenario_best["asn"].unique())
    sns.boxplot(data=samples, whis=boxplot_whis, sym=boxplot_sym, ax=plt.gca(),
                palette= boxplot_palette, **plot_args)
    plt.gca().set_xticklabels(labels)
    
    if boxplot_numerousness: 
        for i, _ in enumerate(plt.gca().get_xticklabels()):

            args = {}
            if boxplot_numerousness_rotate is not None:
                args = {'rotation' : boxplot_numerousness_rotate,
                        'ha' : 'left' if boxplot_numerousness_rotate > 0 else 'right',
                        'va': 'bottom'}

            plt.gca().text(i, 1.05, len(samples[i]), horizontalalignment='center',
                            size=boxplot_numerousness_fontsize,
                            transform = plt.gca().get_xaxis_transform(),
                            **args)
                            
    if boxplot_empty:
        plt.setp(plt.gca().artists, edgecolor = 'k', facecolor='w', linewidth =1)
        plt.setp(plt.gca().lines, color='k', linewidth =1)

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_boxplot_multi(data, path=None, plot_args = {},
         boxplot_sym='', boxplot_whis=[5,95], 
         boxplot_palette=sns.color_palette(), **kw):
        """
        Plot multiple boxplots.

        data: pandas dataframe where each cell is a list or array.
        path (str, optional): path to save the figure.
        plot_args (dict, optional): dict of plot arguments.
        boxplot_sym: symbol of the boxplot.
        boxplot_whis (list[2]<int>, optional): whiskers.
        boxplot_palette (list of color, optional): palette of the boxplot.
        """

        new_data = []
        for c in data:
            for i, l in data[c].iteritems():
                for e in l:
                    new_data.append( {"x":i, "y":e, "hue":c })
        new_data = pd.DataFrame(new_data)
        p = sns.boxplot(x="x", y="y", hue="hue", data=new_data, whis=boxplot_whis, order=data.index,
                    sym=boxplot_sym, ax=plt.gca(), palette= boxplot_palette, **plot_args)
        p.legend().remove()
        plt.xlabel("")
        plt.gca().set_xticklabels(data.index)

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_timeseries(data, path=None, plot_args = {},
        linewidth = 1,timeseries_format='%Y/%m/%d', **kw):
    """
    Plot a timeseries.

    data: series of data with a time index.
    path (str, optional): path to save the figure.
    plot_args (dict, optional): dict of plot arguments.
    linewidth (int, optional): width of the line.
    timeseries_format (str, optional): format of the time index.
    """

    plt.plot(data, markeredgewidth=0, linewidth = linewidth, **plot_args) 
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(timeseries_format))

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_timeseries_multi(data, path, plot_args = {}, linewidth = 1, timeseries_format='%Y/%m/%d', **kw):
    """
    Plot multiple timeseries.

    data: list of tuples (name, series) where series must have a time index.
    path (str, optional): path to save the figure.
    plot_args (dict, optional): dict of plot arguments.
    linewidth (int, optional): width of the line.
    timeseries_format (str, optional): format of the time index.
    """
    for name, series in data:
        plt.plot(series, markeredgewidth=0, label = name, linewidth = linewidth, **plot_args) 
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(timeseries_format))

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_timeseries_stacked(data, path=None, plot_args = {}, timeseries_format='%Y/%m/%d', **kw):
    """
    Plot a stacked timeseries.

    data: pandas dataframe with time index and multiple columns per series.
    path (str, optional): path to save the figure.
    plot_args (dict, optional): dict of plot arguments.
    timeseries_format (str, optional): format of the time index.

    """
    plt.stackplot(data.index,  np.transpose(data.values), lw=0, labels = data.columns, **plot_args)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(timeseries_format))


@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_bars(data, path=None, plot_args={},
         linewidth = 1, bars_width=0.6, **kw):
    yy = [d[1] for d in data]
    xticks_labels_from_data = [d[0] for d in data]
    xx = range(len(yy))
    plt.bar(xx, yy, linewidth = linewidth, align = 'center', width = bars_width, **plot_args)
    plt.xticks(xx, xticks_labels_from_data)
    plt.xlim((-0.5, len(xx) -0.5 )) # Default pretty xlim


@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_bars_multi(data, path=None, plot_args = {},
         linewidth = 1, bars_width=0.6, **kw):
    xticks_labels_from_data = list(data.index)
    num_rows = len(data.index)
    num_columns = len(data.columns)
    bars_width_real=bars_width/num_columns
    prop_iter = iter(plt.rcParams['axes.prop_cycle'])
    for i, column in enumerate( data ):
        delta = -bars_width/2 + i*bars_width_real + bars_width_real/2
        plt.bar( [e + delta for e in range(num_rows)], list(data[column]), linewidth = linewidth,
                align = 'center', width = bars_width_real, label = column,
                color=next(prop_iter)['color'], **plot_args)
    plt.xticks(range(num_rows), xticks_labels_from_data)
    plt.xlim((-0.5, num_rows -0.5 )) # Default pretty xlim

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_bars_stacked(data, path=None, plot_args = {},
        linewidth = 1, bars_width=0.6, **kw):
    xticks_labels_from_data = list(data.index)
    num_rows = len(data.index)
    num_columns = len(data.columns)
    prop_iter = iter(plt.rcParams['axes.prop_cycle'])
    bottom = np.zeros(num_rows)
    for i, column in enumerate( data ):
        plt.bar(range(num_rows), list(data[column]), bottom=bottom, linewidth = linewidth,
                align = 'center', width = bars_width, label = column,
                color=next(prop_iter)['color'], **plot_args)
        bottom = np.add(bottom, list(data[column]))
        
    plt.xticks(range(num_rows), xticks_labels_from_data)
    plt.xlim((-0.5, num_rows -0.5 )) # Default pretty xlim

@backup_mpl_rccontext()
@run_after(configuration_preplot)
@run_before(configuration_postplot)
def plot_callback(plt, callback=None, **kw):
    callback(plt)
