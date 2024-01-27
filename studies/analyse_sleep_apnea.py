import pandas as pd
from matplotlib import pyplot as plt
from fit import helpers, gee, dproc, plotter
import numpy as np
import matplotlib
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore",category =RuntimeWarning)
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
matplotlib.rcParams.update({'font.size': 17})

df=pd.read_csv('../data/presernovi/2023_06_28.csv')
df_results=pd.DataFrame()

factors=['Kortizol','PER1','PER2','PER3','CRY1','CRY2','BMAL1','Melatonin']
methods=['cosinor']
amp_threshold=0.05
for method in methods:
    for factor in factors:
        df_new= helpers.clean_data(df, 'Hour', factor)
        df_new['population_id']=0
        df_pop = df_new.rename(columns={"Hour": "X", factor: "Y"})
        df_pop1=df_pop[df_pop['OSA']==True]
        df_pop2 = df_pop[df_pop['OSA'] == False]

        # NON POP ANAL
        results1, amp1, acr1, statistics1= dproc.non_population_fit_cosinor1(df_pop1)
        results2, amp2, acr2, statistics2 = dproc.non_population_fit_cosinor1(df_pop2)

        p_amp1 = statistics1['p-values'][1]
        p_amp2 = statistics2['p-values'][1]

        print(factor)
        if p_amp1 < amp_threshold:
            print('testna')
        if p_amp2 < amp_threshold:
            print('kontolna')

        print('---------')

        if p_amp1 < amp_threshold or p_amp2<amp_threshold:
            print(factor)
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            result1,ax= dproc.fit_non_population(df_pop1, method, n_components=1, n_periods=1, ax=ax, label='testna', color='darkorange', raw_label='izvorni podatki - testna')
            result2,ax= dproc.fit_non_population(df_pop2, method, n_components=1, n_periods=1, ax=ax, label='kontrolna', color='darkblue', raw_label='izvorni podatki - kontrolna')
            ax.legend()
            plt.title(factor)
            plt.savefig('../results/preseren/nonpop_'+factor+".png")
            plt.show()
            plt.close()

            if type(result1)!=int:
                row1 = {'factor': factor, 'OSA': True, 'pop_anal': False, 'method': method,
                        'amplitude': result1['amplitude'], 'acrophase': result1['acrophase'], 'mesor': result1['mesor']}
                df_results = pd.concat([df_results, (pd.DataFrame.from_dict(row1, orient='index')).T],
                                       ignore_index=True)
            if type(result2)!=int:
                row2 = {'factor': factor, 'OSA': False, 'pop_anal': False, 'method':method,'amplitude': result2['amplitude'],
                    'acrophase': result2['acrophase'], 'mesor': result2['mesor']}
                df_results = pd.concat([df_results, (pd.DataFrame.from_dict(row2, orient='index')).T],
                                       ignore_index=True)

        # POP ANAL
        df_new = df.copy().dropna(subset=['Hour', factor])
        df_new['population_id'] = 0
        df_pop = df_new.rename(columns={"Hour": "X", factor: "Y"})
        df_pop1 = df_pop[df_pop['OSA'] == True]
        df_pop2 = df_pop[df_pop['OSA'] == False]
        df_cosinor1 = dproc.population_fit_cosinor1(df_pop1, 24, save_to='', alpha=0.05, plot_on=False,
                                                    plot_individuals=False,
                                                    plot_measurements=False, plot_margins=False, color="blue",
                                                    hold_on=False,
                                                    plot_residuals=False, save_folder='')

        df_cosinor2 = dproc.population_fit_cosinor1(df_pop2, 24, save_to='', alpha=0.05, plot_on=False,
                                                    plot_individuals=False,
                                                    plot_measurements=False, plot_margins=False, color="blue",
                                                    hold_on=False,
                                                    plot_residuals=False, save_folder='')
        df_cosinor1 = df_cosinor1.iloc[0]
        p_amp1 = df_cosinor1['p(amplitude)']
        df_cosinor2 = df_cosinor2.iloc[0]
        p_amp2 = df_cosinor2['p(amplitude)']

        if p_amp1 < amp_threshold or p_amp2<amp_threshold:
            print(factor)

            # GEE
            if method=='cosinor':
                result_gee= gee.gee_cosinor(df_new, factor, 'Hour', 'id', variables=['OSA'], n_components=1, save_to='../results/preseren/gee_' + factor + ".png", plot_title=(factor))

                if result_gee['parameter'].iloc[0][0]==False:
                    row2 = {'factor': factor, 'OSA': False, 'pop_anal': True, 'method': method,
                            'amplitude': result_gee['amplitude'].iloc[0],
                            'acrophase': result_gee['acrophase'].iloc[0], 'mesor': result_gee['mesor'].iloc[0]}
                    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(row2, orient='index')).T],
                                           ignore_index=True)
                elif result_gee['parameter'].iloc[0][0]==True:
                    row2 = {'factor': factor, 'OSA': True, 'pop_anal': True, 'method': method,
                            'amplitude': result_gee['amplitude'].iloc[0],
                            'acrophase': result_gee['acrophase'].iloc[0], 'mesor': result_gee['mesor'].iloc[0]}
                    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(row2, orient='index')).T],
                                           ignore_index=True)

                if result_gee['parameter'].iloc[1][0] == False:
                    row1 = {'factor': factor, 'OSA': False, 'pop_anal': True, 'method': method,
                            'amplitude': result_gee['amplitude'].iloc[1],
                            'acrophase': result_gee['acrophase'].iloc[1], 'mesor': result_gee['mesor'].iloc[1]}
                    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(row1, orient='index')).T],
                                           ignore_index=True)
                elif result_gee['parameter'].iloc[1][0] == True:
                    row1 = {'factor': factor, 'OSA': True, 'pop_anal': True, 'method': method,
                            'amplitude': result_gee['amplitude'].iloc[1],
                            'acrophase': result_gee['acrophase'].iloc[1], 'mesor': result_gee['mesor'].iloc[1]}
                    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(row1, orient='index')).T],
                                           ignore_index=True)

                #POVPRECENJE IND MODELOV
                row1={'factor': factor, 'OSA': True, 'pop_anal': True, 'method': 'cosinor_1',
                            'amplitude': np.round(df_cosinor1['amplitude'],2),
                            'acrophase': np.round(df_cosinor1['acrophase'],2), 'mesor': np.round(df_cosinor1['mesor'],2)}
                row2 = {'factor': factor, 'OSA': False, 'pop_anal': True, 'method': 'cosinor_1',
                        'amplitude': np.round(df_cosinor2['amplitude'],2),
                            'acrophase': np.round(df_cosinor2['acrophase'],2), 'mesor': np.round(df_cosinor2['mesor'],2)}
                df_results = pd.concat([df_results, (pd.DataFrame.from_dict(row1, orient='index')).T],
                                       ignore_index=True)
                df_results = pd.concat([df_results, (pd.DataFrame.from_dict(row2, orient='index')).T],
                                       ignore_index=True)

                X_test = np.linspace(0, 23, 100)
                y1=df_cosinor1['Y_test']
                y2=df_cosinor2['Y_test']
                fig, ax = plt.subplots(1, 1, figsize=(12, 7))
                plotter.subplot_model(df_pop1['X'].to_numpy(), df_pop1['Y'].to_numpy(), X_test, y1, ax, plot_model=True, plot_measurements=True, plot_measurements_with_color=True, color='darkorange', fit_label='testna', raw_label='izvorni podatki - testna')
                plotter.subplot_model(df_pop2['X'].to_numpy(), df_pop2['Y'].to_numpy(), X_test, y2, ax, plot_model=True, plot_measurements=True, plot_measurements_with_color=True, color='darkblue', fit_label='kontrolna', raw_label='izvorni podatki - kontrolna')
                ax.legend()
                plt.title(factor)
                plt.savefig('../results/preseren/povpr_'+factor+".png")
                plt.show()
                plt.close()


df_results.to_csv('../results/preseren/res.csv',index=False)

