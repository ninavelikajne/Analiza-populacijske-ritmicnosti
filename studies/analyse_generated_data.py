import numpy as np
import pandas as pd
from fit import helpers as hlp, dproc
import ast

colors = ['blue', 'green', 'orange', 'red', 'purple', 'olive', 'tomato', 'yellow', 'pink', 'turquoise', 'lightgreen']

df=pd.read_csv('../data/generated_data.csv')
df_results=pd.DataFrame()
df_separated=pd.DataFrame()
df_periods=pd.DataFrame()
names=['asymmetric_oscillatory_2','symmetric_non_oscillatory','asymmetric_non_oscillatory','symmetric_oscillatory','asymmetric_oscillatory_1']
for name in names:
    print(name)
    df_data = df[df.data == name]

    cosinor1_tp, cosinor1_fp, cosinor1_tn, cosinor1_fn = 0, 0, 0, 0
    cosinor11_tp, cosinor11_fp, cosinor11_tn, cosinor11_fn = 0, 0, 0, 0
    cosinor2_tp, cosinor2_fp, cosinor2_tn, cosinor2_fn = 0, 0, 0, 0
    cosinor3_tp, cosinor3_fp, cosinor3_tn, cosinor3_fn = 0, 0, 0, 0
    cosopt_tp, cosopt_fp, cosopt_tn, cosopt_fn = 0, 0, 0, 0
    arser1_tp, arser1_fp, arser1_tn, arser1_fn = 0, 0, 0, 0
    arser2_tp, arser2_fp, arser2_tn, arser2_fn = 0, 0, 0, 0
    arser3_tp, arser3_fp, arser3_tn, arser3_fn = 0, 0, 0, 0

    #umetno=[134]
    for ix,population_id in enumerate(df_data['population_id'].unique()):#(umetno):#(df_data['population_id'].unique()):
        df_pop = df_data[df_data.population_id == population_id]
        repetition=df_pop['repetition'].iloc[0]
        changing_param=df_pop['changing_param'].iloc[0]
        print(str(population_id)+" rep: "+str(repetition))

        a = df_pop['data_params'].iloc[0]
        b = ast.literal_eval(a)
        time_step = b['step']

        # 1_cosinor1
        df_cosinor1= dproc.population_fit_cosinor1(df_pop, 24, save_to='', alpha=0.05, plot_on=False, plot_individuals=False,
                                                   plot_measurements=False, plot_margins=False, color="blue", hold_on=False,
                                                   plot_residuals=False, save_folder='')
        df_cosinor1 = df_cosinor1.iloc[0]
        tp, fp, tn, fn,classification = hlp.cosinor1_classify_rhythm(df_pop['rhythm'].iloc[0], df_cosinor1['p(amplitude)'])
        cosinor1_tp=cosinor1_tp+tp
        cosinor1_fp=cosinor1_fp+fp
        cosinor1_tn=cosinor1_tn+tn
        cosinor1_fn=cosinor1_fn+fn
        separeated={'method':'cosinor_1','data_name':name,'data_params':a,'changing_param':changing_param,'time_span':b['time_span'], 'num_of_period':b['num_of_period'], 'step':b['step'], 'replicates':b['replicates'], 'noise':b['noise'],'population_id':population_id,'repetition':repetition,'amplitude':df_cosinor1['amplitude'],'acrophase':df_cosinor1['acrophase'],'mesor':df_cosinor1['mesor'],'classification':classification,'Y_test':np.array2string(df_cosinor1['Y_test'])}
        df_separated = pd.concat([df_separated, (pd.DataFrame.from_dict(separeated, orient='index')).T],ignore_index=True)

        # cosinor (n_comp=1)
        df_cosinor11,_ = dproc.fit_population(df_pop, 'cosinor', n_components=1, fig_name=str(population_id) + '_1', plot_population=False, plot_individuals=False)
        df_cosinor11 = df_cosinor11.iloc[0]

        tp, fp, tn, fn,classification=hlp.classify_rhythm(df_pop['rhythm'].iloc[0],df_cosinor11['amplitude'])
        cosinor11_tp=cosinor11_tp+tp
        cosinor11_fp=cosinor11_fp+fp
        cosinor11_tn=cosinor11_tn+tn
        cosinor11_fn=cosinor11_fn+fn
        separeated={'method':'cosinor1','data_name':name,'data_params':a,'changing_param':changing_param,'time_span':b['time_span'], 'num_of_period':b['num_of_period'], 'step':b['step'], 'replicates':b['replicates'], 'noise':b['noise'],'population_id':population_id,'repetition':repetition,'amplitude':df_cosinor11['amplitude'],'acrophase':df_cosinor11['acrophase'],'mesor':df_cosinor11['mesor'],'classification':classification,'Y_test':np.array2string(df_cosinor11['Y_test'])}
        df_separated = pd.concat([df_separated, (pd.DataFrame.from_dict(separeated, orient='index')).T],
                                 ignore_index=True)

        # cosinor2
        df_cosinor2,_ = dproc.fit_population(df_pop, 'cosinor', n_components=2, fig_name=str(population_id) + '_2', plot_population=False, plot_individuals=False)
        df_cosinor2 = df_cosinor2.iloc[0]

        tp, fp, tn, fn,classification=hlp.classify_rhythm(df_pop['rhythm'].iloc[0],df_cosinor2['amplitude'])
        cosinor2_tp=cosinor2_tp+tp
        cosinor2_fp=cosinor2_fp+fp
        cosinor2_tn=cosinor2_tn+tn
        cosinor2_fn=cosinor2_fn+fn
        separeated={'method':'cosinor2','data_name':name,'data_params':a,'changing_param':changing_param,'time_span':b['time_span'], 'num_of_period':b['num_of_period'], 'step':b['step'], 'replicates':b['replicates'], 'noise':b['noise'],'population_id':population_id,'repetition':repetition,'amplitude':df_cosinor2['amplitude'],'acrophase':df_cosinor2['acrophase'],'mesor':df_cosinor2['mesor'],'classification':classification,'Y_test':np.array2string(df_cosinor2['Y_test'])}
        df_separated = pd.concat([df_separated, (pd.DataFrame.from_dict(separeated, orient='index')).T],
                                 ignore_index=True)

        # cosinor3
        df_cosinor3,_ = dproc.fit_population(df_pop, 'cosinor', n_components=3, fig_name=str(population_id) + '_3', plot_population=False, plot_individuals=False)
        df_cosinor3 = df_cosinor3.iloc[0]

        tp, fp, tn, fn,classification = hlp.classify_rhythm(df_pop['rhythm'].iloc[0], df_cosinor3['amplitude'])
        cosinor3_tp = cosinor3_tp + tp
        cosinor3_fp = cosinor3_fp + fp
        cosinor3_tn = cosinor3_tn + tn
        cosinor3_fn = cosinor3_fn + fn
        separeated={'method':'cosinor3','data_name':name,'data_params':a,'changing_param':changing_param,'time_span':b['time_span'], 'num_of_period':b['num_of_period'], 'step':b['step'], 'replicates':b['replicates'], 'noise':b['noise'],'population_id':population_id,'repetition':repetition,'amplitude':df_cosinor3['amplitude'],'acrophase':df_cosinor3['acrophase'],'mesor':df_cosinor3['mesor'],'classification':classification,'Y_test':np.array2string(df_cosinor3['Y_test'])}
        df_separated = pd.concat([df_separated, (pd.DataFrame.from_dict(separeated, orient='index')).T],
                                 ignore_index=True)

        # cosopt
        df_cosopt,_ = dproc.fit_population(df_pop, 'cosopt', fig_name=str(population_id), plot_population=False, plot_individuals=False)
        df_cosopt = df_cosopt.iloc[0]

        tp, fp, tn, fn,classification = hlp.classify_rhythm(df_pop['rhythm'].iloc[0], df_cosopt['amplitude'])
        cosopt_tp = cosopt_tp + tp
        cosopt_fp = cosopt_fp + fp
        cosopt_tn = cosopt_tn + tn
        cosopt_fn = cosopt_fn + fn
        separeated={'method':'cosopt','data_name':name,'data_params':a,'changing_param':changing_param,'time_span':b['time_span'], 'num_of_period':b['num_of_period'], 'step':b['step'], 'replicates':b['replicates'], 'noise':b['noise'],'population_id':population_id,'repetition':repetition,'amplitude':df_cosopt['amplitude'],'acrophase':df_cosopt['acrophase'],'mesor':df_cosopt['mesor'],'classification':classification,'Y_test':np.array2string(df_cosopt['Y_test'])}
        df_separated = pd.concat([df_separated, (pd.DataFrame.from_dict(separeated, orient='index')).T],
                                 ignore_index=True)

        try:
            df_arser1,df_periods = dproc.fit_population(df_pop, 'arser', n_periods=1, time_step=time_step, fig_name=str(population_id) + '_1', plot_population=False, plot_individuals=False, df_periods=df_periods, pop_id=population_id)
            df_arser1 = df_arser1.iloc[0]

            tp, fp, tn, fn, classification = hlp.classify_rhythm(df_pop['rhythm'].iloc[0], df_arser1['amplitude'])
            arser1_tp = arser1_tp + tp
            arser1_fp = arser1_fp + fp
            arser1_tn = arser1_tn + tn
            arser1_fn = arser1_fn + fn
            separeated = {'method': 'arser1', 'data_name': name, 'data_params': a, 'changing_param':changing_param,'time_span': b['time_span'],
                          'num_of_period': b['num_of_period'], 'step': b['step'], 'replicates': b['replicates'],
                          'noise': b['noise'], 'population_id': population_id, 'repetition':repetition,'amplitude': df_arser1['amplitude'],
                          'acrophase': df_arser1['acrophase'], 'mesor': df_arser1['mesor'],
                          'classification': classification, 'Y_test': np.array2string(df_arser1['Y_test'])}

        except Exception as e:
            print("error:")
            print(e)
            print("\t" + str(name) + " " + str(population_id))
            df_arser1 = {'amplitude': 0.0}
            separeated = {'method': 'arser1', 'data_name': name, 'data_params': a,'changing_param':changing_param, 'time_span': b['time_span'],
                          'num_of_period': b['num_of_period'], 'step': b['step'], 'replicates': b['replicates'],
                          'noise': b['noise'], 'population_id': population_id, 'repetition':repetition,'amplitude': 0,
                          'acrophase': 0, 'mesor': 0,
                          'classification': '/', 'Y_test': np.array2string([])}

        df_separated = pd.concat([df_separated, (pd.DataFrame.from_dict(separeated, orient='index')).T],
                                 ignore_index=True)

        # arser2
        try:
            df_arser2,df_periods= dproc.fit_population(df_pop, 'arser', n_periods=2, time_step=time_step, fig_name=str(population_id) + '_2', plot_population=False, plot_individuals=False, df_periods=df_periods, pop_id=population_id)
            df_arser2 = df_arser2.iloc[0]
            tp, fp, tn, fn, classification = hlp.classify_rhythm(df_pop['rhythm'].iloc[0], df_arser2['amplitude'])
            arser2_tp = arser2_tp + tp
            arser2_fp = arser2_fp + fp
            arser2_tn = arser2_tn + tn
            arser2_fn = arser2_fn + fn
            separeated = {'method': 'arser2', 'data_name': name, 'data_params': a, 'changing_param':changing_param,'time_span': b['time_span'],
                          'num_of_period': b['num_of_period'], 'step': b['step'], 'replicates': b['replicates'],
                          'noise': b['noise'], 'population_id': population_id, 'repetition':repetition,'amplitude': df_arser2['amplitude'],
                          'acrophase': df_arser2['acrophase'], 'mesor': df_arser2['mesor'],
                          'classification': classification, 'Y_test': np.array2string(df_arser2['Y_test'])}
        except Exception as e:
            print("error:")
            print(e)
            print("\t" + str(name) + " " + str(population_id))
            separeated = {'method': 'arser2', 'data_name': name, 'data_params': a,'changing_param':changing_param, 'time_span': b['time_span'],
                          'num_of_period': b['num_of_period'], 'step': b['step'], 'replicates': b['replicates'],
                          'noise': b['noise'], 'population_id': population_id, 'repetition':repetition,'amplitude': 0,
                          'acrophase': 0, 'mesor': 0,
                          'classification': '/', 'Y_test': np.array2string([])}

        df_separated = pd.concat([df_separated, (pd.DataFrame.from_dict(separeated, orient='index')).T],
                             ignore_index=True)

        # arser3
        try:
            df_arser3,df_periods = dproc.fit_population(df_pop, 'arser', n_periods=3, time_step=time_step, fig_name=str(population_id) + '_3', plot_population=False, plot_individuals=False, df_periods=df_periods, pop_id=population_id)
            df_arser3 = df_arser3.iloc[0]
            tp, fp, tn, fn, classification = hlp.classify_rhythm(df_pop['rhythm'].iloc[0], df_arser3['amplitude'])
            arser3_tp = arser3_tp + tp
            arser3_fp = arser3_fp + fp
            arser3_tn = arser3_tn + tn
            arser3_fn = arser3_fn + fn
            separeated = {'method': 'arser3', 'data_name': name, 'data_params': a,'changing_param':changing_param, 'time_span': b['time_span'],
                          'num_of_period': b['num_of_period'], 'step': b['step'], 'replicates': b['replicates'],
                          'noise': b['noise'], 'population_id': population_id, 'repetition':repetition,'amplitude': df_arser3['amplitude'],
                          'acrophase': df_arser3['acrophase'], 'mesor': df_arser3['mesor'],
                          'classification': classification, 'Y_test': np.array2string(df_arser3['Y_test'])}

        except Exception as e:
            print("error:")
            print(e)
            print("\t" + str(name) + " " + str(population_id))
            separeated = {'method': 'arser3', 'data_name': name, 'data_params': a,'changing_param':changing_param, 'time_span': b['time_span'],
                          'num_of_period': b['num_of_period'], 'step': b['step'], 'replicates': b['replicates'],
                          'noise': b['noise'], 'population_id': population_id, 'repetition':repetition,'amplitude': 0,
                          'acrophase': 0, 'mesor': 0,
                          'classification': '/', 'Y_test': np.array2string([])}

        df_separated = pd.concat([df_separated, (pd.DataFrame.from_dict(separeated, orient='index')).T],
                                 ignore_index=True)


    cosinor1_result = {'method': 'cosinor_1', 'data': name, 'tp': cosinor1_tp, 'fp': cosinor1_fp, 'tn': cosinor1_tn,'fn': cosinor1_fn}
    cosinor11_result = {'method': 'cosinor1', 'data': name, 'tp': cosinor11_tp, 'fp': cosinor11_fp, 'tn': cosinor11_tn,'fn': cosinor11_fn}
    cosinor2_result = {'method': 'cosinor2', 'data': name, 'tp': cosinor2_tp, 'fp': cosinor2_fp, 'tn': cosinor2_tn,'fn': cosinor2_fn}
    cosinor3_result = {'method': 'cosinor3', 'data': name, 'tp': cosinor3_tp, 'fp': cosinor3_fp, 'tn': cosinor3_tn,'fn': cosinor3_fn}
    cosopt_result = {'method': 'cosopt', 'data': name, 'tp': cosopt_tp, 'fp': cosopt_fp,'tn': cosopt_tn, 'fn': cosopt_fn}
    arser1_result = {'method': 'arser1', 'data': name, 'tp': arser1_tp, 'fp': arser1_fp,'tn': arser1_tn, 'fn': arser1_fn}
    arser2_result = {'method': 'arser2', 'data': name, 'tp': arser2_tp, 'fp': arser2_fp, 'tn': arser2_tn,'fn': arser2_fn}
    arser3_result = {'method': 'arser3', 'data': name, 'tp': arser3_tp, 'fp': arser3_fp, 'tn': arser3_tn,'fn': arser3_fn}

    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(cosinor1_result, orient='index')).T], ignore_index=True)
    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(cosinor11_result, orient='index')).T], ignore_index=True)
    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(cosinor2_result, orient='index')).T], ignore_index=True)
    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(cosinor3_result, orient='index')).T], ignore_index=True)
    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(cosopt_result, orient='index')).T], ignore_index=True)
    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(arser1_result, orient='index')).T], ignore_index=True)
    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(arser2_result, orient='index')).T], ignore_index=True)
    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(arser3_result, orient='index')).T], ignore_index=True)

df_results.to_csv('./results/classification.csv', index=False)
df_separated.to_csv('./results/sep_classification.csv', index=False)
df_periods.to_csv('./results/arser_periods.csv', index=False)

