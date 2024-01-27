import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from fit.helpers import get_scores, param_dict

colors = ['blue', 'orange', 'green', 'red', 'purple', 'saddlebrown', 'mediumvioletred', 'dimgray']
sns.set_palette(sns.color_palette(colors))
hue_order = ['cosopt', 'cosinor_1', 'cosinor1', 'arser1', 'cosinor2', 'arser2', 'cosinor3', 'arser3']

df = pd.read_csv('../results/sep_classification.csv')
agg_results = pd.DataFrame()

# all data
data_name = 'all'
for method in df['method'].unique():
    a = df[df.method == method]
    for repetition in df['repetition'].unique():
        b = a[a.repetition == repetition]

        agg_results = get_scores(b, 'num_of_period', agg_results, method, data_name, repetition=repetition)
        agg_results = get_scores(b, 'step', agg_results, method, data_name, repetition=repetition)
        agg_results = get_scores(b, 'replicates', agg_results, method, data_name, repetition=repetition)
        agg_results = get_scores(b, 'noise', agg_results, method, data_name, repetition=repetition)

agg_results.to_csv('../results/agg_classification.csv', index=False)

# plot
for data_name in ["all"]:
    test = agg_results[agg_results.data_name == data_name]
    metrics = ['mcc']  # ,'f1','precision','recall']
    gs = gridspec.GridSpec(4, 1)
    fig = plt.figure(figsize=(15, 30))
    for metric in metrics:
        for ixic, param in enumerate(['noise', 'num_of_period', 'replicates', 'step']):  # test.param.unique():
            ax = fig.add_subplot(gs[ixic])
            repeats = 4
            if param == 'noise' or param == 'replicates':
                repeats = 3
            a = test[test.param == param]
            snsFig = sns.boxplot(data=a, x='param_val', y=metric, hue='method', ax=ax, fill=False, hue_order=hue_order)

            for i, box in enumerate([p for p in snsFig.artists if not p.get_label()]):
                color_ix = i // repeats
                color = colors[color_ix]
                box.set_edgecolor(color)
                box.set_facecolor((0, 0, 0, 0))
                # iterate over whiskers and median lines
                for j in range(5 * i, 5 * (i + 1)):
                    snsFig.lines[j].set_color(color)

            handles, labels = snsFig.get_legend_handles_labels()
            snsFig = sns.stripplot(data=a, x='param_val', y=metric, hue='method',
                                   dodge=True, ax=snsFig)

            if param == 'num_of_period':
                snsFig.legend(handles, labels, loc='lower right')
            else:
                snsFig.legend(handles, labels)

            ax.set_xlabel(param_dict[param], fontsize=25)
            ax.set_ylabel('MCC', fontsize=25)

plt.savefig('../results/classification/' + data_name + "_" + metric + ".pdf", bbox_inches='tight')
plt.show()
