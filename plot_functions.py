import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce

def simple_plot(X,Y,**kwargs):
    plt.semilogx(X,Y, **kwargs)

def multiplots(plots, shadow, average, logx = True):
    legend_items = 0
    for p in plots.values():
        plt.plot(p[0].X,p[0].Y,**p[1],label=p[2])
        legend_items += 1

    if average or shadow:
        dfs = [p[0] for p in plots.values() if p[3] == "WT"]
        df_final = reduce(lambda left, right: pd.merge(left, right, on='X',how="outer"), dfs)
        X = df_final.X
        df_final.drop(columns=["X"],inplace=True)

        Ymax = df_final.apply(np.nanmax,axis=1).values
        Ymin = df_final.apply(np.nanmin, axis=1).values
        Yav = df_final.apply(np.nanmedian, axis=1).values

    if average:
        plt.plot(X, Yav, color="black", ls="--", linewidth=2, label="Median")
        legend_items += 1
    #plt.plot(X, Ymax, color="black", linewidth=4, legend="Min/Max")
    #plt.plot(X, Ymin, color="black", linewidth=4)

    if shadow:
        plt.fill_between(X,Ymin,Ymax,alpha=0.1)

    plt.xscale("log")

    # Shrink current axis's height by 10% on the bottom
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=(legend_items + 1) // 3)


def multiplot_with_subplots(plots, xlabel, y_label):
    ncols = 3
    nrows = len(plots.values()) // ncols
    if ncols*nrows < len(plots.values()):
        nrows += 1
    for ind, p in enumerate(plots.values()):
        plt.subplot(nrows, ncols, ind+1)
        plt.plot(p[0].X,p[0].Y,**p[1],label=p[2])
        plt.xscale("log")
        plt.gca().tick_params(axis='both', which='major', labelsize=6)
        plt.gca().tick_params(axis='both', which='minor', labelsize=5)
        # Shrink current axis's height by 10% on the bottom
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize="xx-small")
        plt.xlabel(xlabel, fontdict={"size":"x-small"})
        plt.ylabel(y_label, fontdict={"size":"x-small"})