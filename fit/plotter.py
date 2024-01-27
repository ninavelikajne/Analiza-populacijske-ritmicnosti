import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

colors = ['darkblue', 'darkorange', 'darkgreen', 'darkred', 'purple', 'saddlebrown', 'mediumvioletred', 'dimgray']


def plot_single(data, results, test='', plot_measurements=True, save_to='', plot_dense=True, plot_margins=True,
                period=24):
    X = data.x.values
    Y = data.y.values

    color = "black"

    if plot_dense:
        X_fit = np.linspace(min(X), max(X), 1000)

        rrr_fit = np.cos(2 * np.pi * X_fit / period)
        sss_fit = np.sin(2 * np.pi * X_fit / period)

        data = pd.DataFrame()
        data['rrr'] = rrr_fit
        data['sss'] = sss_fit

        Y_fit = results.predict(data)

        D = np.column_stack((rrr_fit, sss_fit))
        D = sm.add_constant(D, has_constant='add')

        if plot_margins:
            _, lower, upper = wls_prediction_std(results, exog=D, alpha=0.05)
            plt.fill_between(X_fit, lower, upper, color=color, alpha=0.1)


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
    return ax


def edit_box_plot(snsFig, df, param, metric, hue, ax, fig, title='', x_label='', y_label='', save=''):
    for i, box in enumerate([p for p in snsFig.artists if not p.get_label()]):
        color = colors[i]
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


def plot_confidence_interval(x, values, ax, value, z=1.96, color='#2187bb', horizontal_line_width=0.25,
                             xtick_rotation=75):
    if value != 'acrophase':
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
    else:
        mean = stats.circmean(list(values), low=-2 * np.pi, high=0)
        stdev = stats.circstd(list(values), low=-2 * np.pi, high=0)
    confidence_interval = z * stdev / np.sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    ax.plot([x, x], [top, bottom], color=color)
    ax.plot([left, right], [top, top], color=color)
    ax.plot([left, right], [bottom, bottom], color=color)
    ax.plot(x, mean, 'o', color='#f44336', markersize=2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(xtick_rotation)

    return mean, confidence_interval
