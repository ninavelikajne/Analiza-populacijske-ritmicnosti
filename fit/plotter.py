import statistics

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np
import matplotlib
import scipy.stats as stats
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
colors=['darkblue','darkorange','darkgreen','darkred','purple','saddlebrown','mediumvioletred','dimgray']



def plot_single(data, results, test='', plot_measurements=True, save_to='', plot_dense=True, plot_margins=True,
                period=24):
    X = data.x.values
    Y = data.y.values

    # if 'control' in test.lower():
    #    color = "black"
    # else:
    #    color = "blue"

    color = "black"

    if plot_dense:
        X_fit = np.linspace(min(X), max(X), 1000)

        rrr_fit = np.cos(2 * np.pi * X_fit / period)
        sss_fit = np.sin(2 * np.pi * X_fit / period)

        data = pd.DataFrame()

        # data['x'] = X_fit
        data['rrr'] = rrr_fit
        data['sss'] = sss_fit
        # data['Intercept'] = 1

        Y_fit = results.predict(data)

        D = np.column_stack((rrr_fit, sss_fit))
        D = sm.add_constant(D, has_constant='add')

        if plot_margins:
            _, lower, upper = wls_prediction_std(results, exog=D, alpha=0.05)
            plt.fill_between(X_fit, lower, upper, color=color, alpha=0.1)
            # plt.fill_between(X_fit, lower, upper, color='#888888', alpha=0.1)


    else:
        X_fit = X
        Y_fit = results.fittedvalues

    L1 = list(zip(X_fit, Y_fit))
    L1.sort()
    X_fit, Y_fit = list(zip(*L1))

    if plot_measurements:
        plt.plot(X, Y, 'o', markersize=1, color=color)

    plt.plot(X_fit, Y_fit, color=color, label=test)

    if plot_measurements:
        plt.axis([min(X), max(X), 0.9 * min(min(Y), min(Y_fit)), 1.1 * max(max(Y), max(Y_fit))])
    else:
        plt.axis([min(X_fit), 1.1 * max(X), min(Y_fit) * 0.9, max(Y_fit) * 1.1])

    plt.title(test + ', p-value=' + "{0:.5f}".format(results.f_pvalue))

    if save_to:
        plt.savefig(save_to + '.pdf')
        plt.savefig(save_to + '.png')
        plt.close()
    else:
        plt.show()


def subplot_model(X, Y, X_test, Y_test, ax, plot_measurements=True, plot_measurements_with_color=False, plot_model=True,
                  title='', color='black', fit_label='', period=24, raw_label='raw data'):
    ax.set_title(title)
    ax.set_xlabel('čas')
    ax.set_ylabel('vrednost')

    if plot_measurements:
        if plot_measurements_with_color:
            ax.plot(X, Y, '.', markersize=1, color=color, label=raw_label)
        else:
            ax.plot(X, Y, '.', markersize=1, color='black', label=raw_label)
    if plot_model:
        ax.plot(X_test, Y_test, label=fit_label, color=color)

    ax.set_xlim(0, period - 1)
    #ax.set_ylim(min(Y.min(),Y_test.min())-1, max(Y.max(),Y_test.max())+1)

    return ax

def edit_box_plot(snsFig,df,param,metric,hue,ax,fig,title='',x_label='',y_label='',save=''):
    for i, box in enumerate([p for p in snsFig.artists if not p.get_label()]):
        color=colors[i]
        box.set_edgecolor(color)
        box.set_facecolor((0, 0, 0, 0))
        # iterate over whiskers and median lines
        for j in range(5 * i, 5 * (i + 1)):
            snsFig.lines[j].set_color(color)

    handles, labels = snsFig.get_legend_handles_labels()
    snsFig = sns.stripplot(data=df, x=param, y=metric, hue=hue,
                           dodge=True, palette="bright", ax=snsFig)
    snsFig.legend(handles, labels, loc='upper right')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig(save)

def plot_confidence_interval(x, values, ax,value,z=1.96,color='#2187bb', horizontal_line_width=0.25,xtick_rotation=75):
    if value!='acrophase':
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
    else:
        mean = stats.circmean(list(values), low=-2*np.pi, high=0)
        stdev = stats.circstd(list(values), low=-2*np.pi, high=0)
    confidence_interval = z * stdev / np.sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    ax.plot([x, x], [top, bottom], color=color)
    ax.plot([left, right], [top, top], color=color)
    ax.plot([left, right], [bottom, bottom], color=color)
    ax.plot(x, mean, 'o', color='#f44336',markersize=2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(xtick_rotation)

    return mean, confidence_interval

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("parameter")
    ax.set_ylabel("method")

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts