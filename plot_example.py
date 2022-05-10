#!/usr/bin/env python3

import fastplot
import numpy as np
import pandas as pd

STYLE="latex"


x = range(11)
y=[4,150,234,465,745,612,554,43,565,987,154]
fastplot.plot((x, y),  'examples/1_line.png', xlabel = 'X', ylabel = 'Y')




x = range(11)
y1=[4,150,234,465,645,612,554,43,565,987,154]
y2=[434,15,24,556,75,345,54,443,56,97,854]
fastplot.plot([ ('First', (x, y1) ), ('Second', (x, y2) )], 'examples/2_line_multi.png',
              mode='line_multi', xlabel = 'X', ylabel = 'Y', xlim = (-0.5,10.5),
              cycler = fastplot.CYCLER_LINESPOINTS, legend=True, legend_loc='upper left',
              legend_ncol=2)




fastplot.plot(np.random.normal(100, 30, 1000), 'examples/3_CDF.png', mode='CDF',
              xlabel = 'Data', style=STYLE)


fastplot.plot(np.random.normal(100, 30, 1000), 'examples/3b_CCDF.png', mode='CDF', 
              CDF_complementary=True, xlabel = 'Data', style=STYLE)


data = [ ('A', np.random.normal(100, 30, 1000)), ('B', np.random.normal(140, 50, 1000)) ]
plot_args={"markevery": [500]}
fastplot.plot(data , 'examples/4_CDF_multi.png', mode='CDF_multi', xlabel = 'Data', legend=True,
              cycler = fastplot.CYCLER_LINESPOINTS, plot_args=plot_args)



data=[ ('A', np.random.normal(100, 30, 450)),
       ('B', np.random.normal(140, 50, 50)),
       ('C', np.random.normal(140, 50, 200))]
fastplot.plot( data,  'examples/5_boxplot.png', mode='boxplot', ylabel = 'Value',
               boxplot_numerousness=True, boxplot_empty=True, boxplot_numerousness_rotate=90)


data = pd.DataFrame(data=[ [np.random.normal(100, 30, 50),np.random.normal(110, 30, 50)],
                           [np.random.normal(90, 30, 50),np.random.normal(90, 30, 50)],
                           [np.random.normal(90, 30, 50),np.random.normal(80, 30, 50)],
                           [np.random.normal(80, 30, 50),np.random.normal(80, 30, 50)]],
                    columns=["Male","Female"], index = ["IT", "FR", "DE", "UK"] )

fastplot.plot( data,  'examples/5b_boxplot_multi.png', mode='boxplot_multi', ylabel = 'Value',
               boxplot_palette="muted", legend=True, legend_ncol=2, ylim=(0,None))



rng = pd.date_range('1/1/2011', periods=480, freq='H')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
fastplot.plot(ts ,  'examples/6_timeseries.png', mode='timeseries', ylabel = 'Value',
              style=STYLE, xticks_rotate=30, xticks_fontsize='small',
              xlim=(pd.Timestamp('1/1/2011'), pd.Timestamp('1/7/2011')))





rng = pd.date_range('1/1/2011', periods=480, freq='H')
ts = pd.Series(np.random.randn(len(rng)), index=rng) + 5
ts2 = pd.Series(np.random.randn(len(rng)), index=rng) + 10
fastplot.plot( [('One', ts), ('Two', ts2)] , 'examples/7_timeseries_multi.png',
               mode='timeseries_multi', ylabel = 'Value', xticks_rotate=30,
               legend = True, legend_loc='upper center', legend_ncol=2, legend_frameon=False,
               ylim = (0,None), xticks_fontsize='small')





rng = pd.date_range('1/1/2011', periods=480, freq='H')
df = pd.DataFrame(np.random.uniform(3,4,size=(len(rng),2)), index=rng, columns=('One','Two'))
df = df.divide(df.sum(axis=1), axis=0)*100
fastplot.plot( df , 'examples/8_timeseries_stacked.png', mode='timeseries_stacked',
               ylabel = 'Value [%]', xticks_rotate=30, ylim=(0,100), legend=True,
               xlim=(pd.Timestamp('1/1/2011'), pd.Timestamp('1/7/2011')))




data = [('First',3),('Second',2),('Third',7),('Four',6),('Five',5),('Six',4)]
fastplot.plot(data,  'examples/9_bars.png', mode = 'bars', ylabel = 'Value',
              xticks_rotate=30, style='serif', ylim = (0,10))





data = pd.DataFrame( [[2,5,9], [3,5,7], [1,6,9], [3,6,8], [2,6,8]],
                     index = ['One', 'Two', 'Three', 'Four', 'Five'],
                     columns = ['A', 'B', 'C'] )
fastplot.plot(data,  'examples/10_bars_multi.png', mode = 'bars_multi', style=STYLE,
              ylabel = 'Value', legend = True, ylim = (0,12), legend_ncol=3,
              legend_args={'markerfirst' : False})



data = pd.DataFrame( [[2,5,9], [3,5,7], [1,6,9], [3,6,3], [2,6,2]],
                     index = ['One', 'Two', 'Three', 'Four', 'Five'],
                     columns = ['A', 'B', 'C'] )
fastplot.plot(data,  'examples/12_bars_stacked.png', mode = 'bars_stacked', style='serif',
              ylabel = 'Value', legend = True, xtick_length=0, legend_ncol=3, ylim = (0,25))



x = range(11)
y=[120,150,234,465,745,612,554,234,565,888,154]
def my_callback(plt):
    plt.bar(x,y)
fastplot.plot(None,  'examples/11_callback.png', mode = 'callback', callback = my_callback,
              style=STYLE, xlim=(-0.5, 11.5), ylim=(0, 1000))



data = [ ('A', np.random.chisquare(2, 1000)), ('B', np.random.chisquare(8, 1000)) ]
data = fastplot.lorenz_gini_multi(data)
fastplot.plot(data, 'examples/13_lorenz.png', mode='line_multi', legend=True, grid=True,
              xlabel = 'Samples [%]', ylabel = 'Share [%]', xlim=(0,1), ylim=(0,1))


import seaborn as sns
data = pd.DataFrame([(4,3),(5,4),(4,5),(8,6),(10,8),(3,1),(13,10),(9,7),(11,11)], columns=["x","y"])
def my_callback(plt):
     sns.regplot(x="x", y="y", data=data, ax=plt.gca())
fastplot.plot(None,  'examples/14_seaborn.png', mode = 'callback', callback = my_callback,
              style=STYLE, grid=True)











