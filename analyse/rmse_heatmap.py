import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(font_scale=1.7)

colors = ['darkblue', 'darkorange', 'darkgreen', 'darkred', 'purple', 'saddlebrown', 'mediumvioletred', 'dimgray']

x_heat = ['n:0.3', 'n:1.5', 'n:4.0', 'p:1', 'p:2', 'p:4', 'p:10', 's:1', 's:2', 's:4', 's:8', 'r:5', 'r:10', 'r:50']
y_heat = ['cosopt', 'cosinor_1', 'cosinor1', 'arser1', 'cosinor2', 'arser2', 'cosinor3', 'arser3']
measure_dict = {
    'amplitude_rmse': 'amplituda RMSE',
    'acrophase_rmse': 'faza RMSE',
    'mesor_rmse': 'MESOR RMSE'
}
data_dict = {
    'symmetric_oscillatory': 'simetrični ritmični',
    'asymmetric_oscillatory_1': '1.serija asimetrični ritmični',
    'asymmetric_oscillatory_2': '2.serija asimetrični rimični'
}

df = pd.read_csv('../results/rmse.csv')

gs = gridspec.GridSpec(3, 3)
fig = plt.figure(figsize=(20, 20))

# by data
params = ['noise', 'num_of_period', 'step', 'replicates']
plot_metrics = ['amplitude_rmse', 'acrophase_rmse', 'mesor_rmse']
methods = ['cosopt', 'cosinor_1', 'cosinor1', 'arser1', 'cosinor2', 'arser2', 'cosinor3', 'arser3']
for i, plot_metric in enumerate(plot_metrics):
    for j, data_name in enumerate(df.data_name.unique()):
        f = df[df.data_name == data_name]
        data = -1
        for ix, method in enumerate(methods):
            newrow = []
            for param in params:
                values = f[param].unique()
                values.sort()
                for value in values:
                    a = f[f.method == method]
                    c = a[a.changing_param == param]
                    b = c[c[param] == value]
                    newrow.append(b[plot_metric].median())

            if type(data) == int:
                data = [newrow]
            else:
                data = np.append(data, [newrow], axis=0)

        ax = fig.add_subplot(gs[j, i])
        sns.heatmap(data, ax=ax, xticklabels=x_heat, yticklabels=y_heat,
                    cmap=sns.cubehelix_palette(as_cmap=True))
        if j == 0:
            ax.set_title(measure_dict[plot_metric], fontsize=25)

        if i == 0:
            ax.set_ylabel(data_dict[data_name], fontsize=25)
        if j == 2 and i == 1:
            ax.set_xlabel("\nparameteri generiranja", fontsize=25)
fig.tight_layout()
plt.savefig('../results/rmse/heatmap.pdf')

plt.show()
