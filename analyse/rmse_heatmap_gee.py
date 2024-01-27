import pandas as pd
import  numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

sns.set(font_scale=1.7)
colors=['darkblue','darkgreen','darkred','purple','saddlebrown','mediumvioletred','dimgray'] #darkorange

x_heat=['n:0.3','n:1.5','n:4.0','p:1','p:2','p:4','p:10','s:1','s:2','s:4','s:8','r:5','r:10','r:50']
y_heat=['cosopt','cosinor1','arser1','cosinor2','arser2','cosinor3','arser3'] #cosinor_1'
measure_dict={
    'amplitude_rmse':'amplituda RMSE',
    'acrophase_rmse':'faza RMSE',
    'mesor_rmse': 'MESOR RMSE'
}
data_dict={
    'symmetric_oscillatory':'simetrični ritmični',
    'asymmetric_oscillatory_1':'1.serija asimetrični ritmični',
    'asymmetric_oscillatory_2':'2.serija asimetrični rimični'
}

df=pd.read_csv('../results/rmse_gee.csv')

# all data
# params=['noise','num_of_period','step','replicates']
# plot_metrics=['amplitude_rmse','acrophase_rmse','mesor_rmse']
# for plot_metric in plot_metrics:
#     data = -1
#     for ix,method in enumerate(df.method.unique()):
#         newrow=[]
#         for param in params:
#             values = df[param].unique()
#             values.sort()
#             for value in values:
#                 a = df[df.method == method]
#                 c= a[a.changing_param==param]
#                 b=c[c[param]==value]
#                 newrow.append(b[plot_metric].median()) #todo acrophase circmean
#         if type(data) == int:
#             data=[newrow]
#         else:
#             data=np.append(data,[newrow],axis=0)
#
#
#     fig, ax = plt.subplots()
#     sns.heatmap(data,ax=ax,xticklabels=x_heat, yticklabels=y_heat,cmap=sns.cubehelix_palette(as_cmap=True))#,annot=True, fmt=".1f")
#     ax.set_title('Data: All, '+measure_dict[plot_metric])
#     ax.set_xlabel("Parameter")
#     ax.set_ylabel("Method")
#     fig.tight_layout()
#     plt.savefig('../results/gee/rmse/all/heatmap_'+plot_metric+"_.png")
#plt.show()

gs = gridspec.GridSpec(3, 3)
fig = plt.figure(figsize=(20,20))

# by data
params=['noise','num_of_period','step','replicates']
plot_metrics=['amplitude_rmse','acrophase_rmse','mesor_rmse']
methods=['cosopt','cosinor1','arser1','cosinor2','arser2','cosinor3','arser3']
for i,plot_metric in enumerate(plot_metrics):
    for j,data_name in enumerate(df.data_name.unique()):
        f=df[df.data_name==data_name]
        data = -1
        for ix,method in enumerate(methods):
            newrow=[]
            for param in params:
                values = f[param].unique()
                values.sort()
                for value in values:
                    a = f[f.method == method]
                    c= a[a.changing_param==param]
                    b=c[c[param]==value]

                    med=b[plot_metric].median()
                    if method == "cosinor3" and param=='step' and (value == 4 or value == 8):
                        med=np.nan

                    if method == "arser3" and param=='step' and (value == 8):
                        med=np.nan

                    if method == "arser2" and param=='step' and (value == 8):
                        med=np.nan

                    newrow.append(med)

                    # try:
                    #     if method=="cosinor3" and (value==4 or value==8):
                    #         print(data_name + " " + method + " " + param + " " + str(value))
                    #         fig, ax = plt.subplots(1, 1, figsize=(12, 7))
                    #         X_test = np.linspace(0, 100, 1000)
                    #         if data_name == 'symmetric_oscillatory':
                    #             y_og = st.og_symmetric_oscillatory(X_test, period1=24, period2=24, A1=3, A2=3)
                    #         elif data_name == 'asymmetric_oscillatory_1':
                    #             y_og = st.og_symmetric_oscillatory(X_test, period1=24, period2=12, A1=3, A2=3)
                    #         elif data_name == 'asymmetric_oscillatory_2':
                    #             y_og = st.og_symmetric_oscillatory(X_test, period1=6, period2=8, A1=3, A2=2)
                    #         colors = ['blue', 'orange', 'green', 'red', 'purple', 'saddlebrown', 'mediumvioletred',
                    #                   'dimgray', 'tomato',
                    #                   'pink']
                    #         plotter.subplot_model(X_test, y_og, X_test, y_og, ax, plot_model=True, plot_measurements=False,color='black')
                    #         cnt = 0
                    #         for i_row, rowi in b.iterrows():
                    #             yis = rowi['Y_test'].split('[')[1].split(']')[0]
                    #             yis = np.fromstring(yis, sep=' ')
                    #             plotter.subplot_model(X_test, yis, X_test, yis, ax, plot_model=True, color=colors[cnt])
                    #             cnt = cnt + 1
                    #
                    #         plt.show()
                    # except:
                    #     continue

            if type(data) == int:
                data=[newrow]
            else:
                data=np.append(data,[newrow],axis=0)

        ax = fig.add_subplot(gs[j, i])
        sns.heatmap(data, ax=ax, xticklabels=x_heat, yticklabels=y_heat,
                    cmap=sns.cubehelix_palette(as_cmap=True))  # ,annot=True, fmt=".1f")
        if j == 0 and i == 1:
            ax.set_title(measure_dict[plot_metric] + "\n" + data_dict[data_name], fontsize=25)
        elif j == 0:
            ax.set_title(measure_dict[plot_metric] + "\n", fontsize=25)
        elif i == 1:
            ax.set_title(data_dict[data_name], fontsize=25)

        if i == 0 and j == 1:
            ax.set_ylabel("ritmične metode", fontsize=25)
        if j == 2 and i == 1:
            ax.set_xlabel("parameteri generiranja", fontsize=25)
fig.tight_layout()
plt.savefig('../results/gee/rmse/heatmap.pdf')

plt.show()
