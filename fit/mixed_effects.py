from fit import helpers,methods,dproc,plotter
import ast
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import itertools
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
colors=['darkblue','darkorange','darkgreen','darkred','purple','saddlebrown','mediumvioletred','dimgray']

dicti={'amplitude':'amplituda','acrophase':'faza','mesor':'MESOR'}

def calculate_confidence_intervals_parameters(df, predict_var, time_var, group_var, variables=None,interactions=None,random_var="1",repetitions=20, period=24, n_components=1, plot=True, save_to=''):

    if interactions!=None:
        combos = []
        for var in interactions:
            df = df[~df[var].isna()]
            combos.append(df[var].unique())
        combos = list(itertools.product(*combos))
    elif variables!=None:
        combos = []
        for var in variables:
            df = df[~df[var].isna()]
            combos.append(df[var].unique())
        combos = list(itertools.product(*combos))
    else:
        print("Interactions or variables can't be None.")
        return

    df_results=pd.DataFrame()
    replicates=len(df['participant_id'].unique())
    sample_size = round(replicates - replicates / 3)
    for i in range(0, repetitions):
        sample = helpers.sample_by_id(df, group_var, sample_size)
        df_result = mixed_effects(sample, predict_var, time_var, group_var, interactions=interactions,variables=variables,random_var=random_var,n_components=n_components, period=period)
        df_results = pd.concat([df_results, df_result], ignore_index=True)

    if plot:
        gs = gridspec.GridSpec(1, 3)
        plot_value=['amplitude','acrophase','mesor']
        for i in range(3):
            ax = pl.subplot(gs[0, i])
            str_combos=[helpers.tup2str(x) for x in combos]
            ax.set_xticks(range(len(combos)), str_combos)
            if i==0:
                plt.ylabel("povprečje (CI 95%)")
            for ix,combo in enumerate(combos):
                a=df_results[df_results.parameter==combo]
                plotter.plot_confidence_interval(ix, a[plot_value[i]].to_numpy(), ax, plot_value[i])
            ax.set_title(dicti[plot_value[i]])
            if i==1:
                plt.xlabel("spremenljivke")
        plt.tight_layout()
        plt.savefig(save_to)
        plt.show()

    mean_amplitude = df_results.groupby(['parameter']).amplitude.mean()
    std_amplitude = df_results.groupby(['parameter']).amplitude.std()
    mean_mesor = df_results.groupby(['parameter']).mesor.mean()
    std_mesor = df_results.groupby(['parameter']).mesor.std()

    df_cis=pd.DataFrame()
    for combo in combos:
        mean_amp=mean_amplitude.loc[[combo]].iloc[0]
        std_amp=std_amplitude.loc[[combo]].iloc[0]
        mean_acro,std_acro= helpers.acro_circ(df_results, combo)
        mean_me=mean_mesor.loc[[combo]].iloc[0]
        std_me=std_mesor.loc[[combo]].iloc[0]
        amplitude = np.around(np.array([mean_amp - 1.96 * std_amp, mean_amp + 1.96 * std_amp]), decimals=2)
        acrophase = np.around(np.array([mean_acro - 1.96 * std_acro, mean_acro + 1.96 * std_acro]), decimals=2)
        mesor = np.around(np.array([mean_me - 1.96 * std_me, mean_me + 1.96 * std_me]), decimals=2)
        dict={'parameter': helpers.tup2str(combo), 'amplitude':round(mean_amp, 2), 'std_amplitude':round(std_amp, 2), 'amplitude_CIs':amplitude,
              'acrophase':round(mean_acro,2),'std_acrophase':round(std_acro,2),'acrophase_CIs':acrophase,
              'mesor':round(mean_me,2),'std_mesor':round(std_me,2),'mesor_CIs':mesor}
        df_cis = pd.concat([df_cis, (pd.DataFrame.from_dict(dict, orient='index')).T],
                               ignore_index=True)

    return df_cis

def mixed_effects(df, predict_var, time_var, group_var,random_var="1", variables=None,interactions=None, n_components=1,period=24):
    df_results=pd.DataFrame()

    df=df.reset_index(drop=True)
    columns=helpers.get_column_names(n_components)
    x = df[time_var].to_numpy()

    X_fit, X_test, X_fit_test, X_fit_eval_params = methods.cosinor(x, n_components,period=period)

    df_new = pd.concat([df, pd.DataFrame(X_fit, columns=columns)], axis=1)
    df_new = df_new[~df_new[predict_var].isna()]

    if interactions!=None:
        for var in interactions:
            df_new = df_new[~df_new[var].isna()]
        formula=predict_var+helpers.formulate_formula_interaction(interactions,columns)

        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        combos = []
        for var in interactions:
            combos.append(df_new[var].unique())

        combos = list(itertools.product(*combos))
        for i, combo in enumerate(combos):
            temp = pd.DataFrame(X_fit_test, columns=columns)
            for ix in range(len(combo)):
                temp[interactions[ix]] = combo[ix]
            temp_vals = mdf.predict(temp)
            temp['predicted'] = temp_vals
            rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
            rhythm_params['parameter'] = combo
            df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],
                                   ignore_index=True)
    elif variables!=None:
        for var in variables:
            df_new = df_new[~df_new[var].isna()]
        formula = predict_var + helpers.formulate_formula(variables, columns)

        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        combos = []
        for var in variables:
            combos.append(df_new[var].unique())

        combos = list(itertools.product(*combos))
        for i, combo in enumerate(combos):
            temp = pd.DataFrame(X_fit_test, columns=columns)
            measurements_df=df_new.copy()
            for ix in range(len(combo)):
                temp[variables[ix]] = combo[ix]
                measurements_df=measurements_df[measurements_df[variables[ix]]==combo[ix]]
            temp_vals = mdf.predict(temp)
            temp['predicted'] = temp_vals
            rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
            rhythm_params['parameter'] = combo
            df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],
                                   ignore_index=True)
    else:
        print("Interactions or variables can't be None.")
        return

    return df_results

def me_cosinor(df, predict_var, time_var, group_var, variables=None,interactions=None, n_components=1,period=24, random_var="1", plot=True,save_to='',plot_title='',summary=False):
    df_results=pd.DataFrame()
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    df=df.reset_index(drop=True)
    columns= helpers.get_column_names(n_components)
    x = df[time_var].to_numpy()

    X_fit, X_test, X_fit_test, X_fit_eval_params = methods.cosinor(x, n_components, period=period)

    df_new = pd.concat([df, pd.DataFrame(X_fit, columns=columns)], axis=1)
    df_new = df_new[~df_new[predict_var].isna()]
    if random_var != "1":
        df_new = df_new[~df_new[random_var].isna()]

    if interactions == None and variables == None:
        formula = predict_var + "~"
        for i, col in enumerate(columns):
            if i == 0:
                formula = formula + col
            else:
                formula = formula + "+" + col

        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        if summary:
            print(mdf.summary())

        temp = pd.DataFrame(X_fit_test, columns=columns)
        temp_vals = mdf.predict(temp)
        temp['predicted'] = temp_vals
        rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
        rhythm_params['Y_test'] = temp_vals
        df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],
                               ignore_index=True)
        if plot:
            plotter.subplot_model(X_test, temp['predicted'], X_test, temp['predicted'], ax, plot_model=True,
                                  plot_measurements=False, color=colors[0])
            plotter.subplot_model(df_new[time_var], df_new[predict_var],
                                  df_new[time_var], df_new[predict_var], ax, plot_model=False,
                                  plot_measurements=True, color=colors[0],
                                  plot_measurements_with_color=True)

    elif interactions!=None:
        for var in interactions:
            df_new = df_new[~df_new[var].isna()]
        formula= predict_var + helpers.formulate_formula_interaction(interactions, columns)

        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        if summary:
            print(mdf.summary())

        combos = []
        for var in interactions:
            combos.append(df_new[var].unique())

        combos = list(itertools.product(*combos))
        for i, combo in enumerate(combos):
            temp = pd.DataFrame(X_fit_test, columns=columns)
            for ix in range(len(combo)):
                temp[interactions[ix]] = combo[ix]
            temp_vals = mdf.predict(temp)
            temp['predicted'] = temp_vals
            rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
            rhythm_params['parameter'] = combo
            df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],
                                   ignore_index=True)
            if plot:
                plotter.subplot_model(X_test, temp['predicted'], X_test, temp['predicted'], ax, plot_model=True,
                                      plot_measurements=False, fit_label=str(combo), color=colors[i])
    elif variables!=None:
        for var in variables:
            df_new = df_new[~df_new[var].isna()]
        formula = predict_var + helpers.formulate_formula(variables, columns)

        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        if summary:
            print(mdf.summary())

        combos = []
        for var in variables:
            combos.append(df_new[var].unique())

        combos = list(itertools.product(*combos))
        for i, combo in enumerate(combos):
            temp = pd.DataFrame(X_fit_test, columns=columns)
            measurements_df=df_new.copy()
            for ix in range(len(combo)):
                temp[variables[ix]] = combo[ix]
                measurements_df=measurements_df[measurements_df[variables[ix]]==combo[ix]]
            temp_vals = mdf.predict(temp)
            temp['predicted'] = temp_vals
            rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
            rhythm_params['parameter'] = combo
            df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],
                                   ignore_index=True)
            if plot:
                plotter.subplot_model(X_test, temp['predicted'], X_test, temp['predicted'], ax, plot_model=True,
                                      plot_measurements=False, fit_label=str(combo), color=colors[i])
    else:
        print("Interactions or variables can't be None.")
        return

    if plot:
        plt.title(plot_title)
        ax.legend()
        fig.savefig(save_to)
        plt.show()
    return df_results



def me_cosopt(df, predict_var, time_var, group_var, variables=None,interactions=None,period=24,random_var="1", plot=True,save_to='',plot_title='',summary=False):
    df_results=pd.DataFrame()
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    df=df.reset_index(drop=True)
    columns= helpers.get_column_names(1)

    df_new = -1
    phases=[]
    for id in df[group_var].unique():
        temp=df[df[group_var]==id].copy()
        temp_X_fit, temp_X_test, temp_X_fit_test, temp_X_fit_eval_params, temp_phase= methods.cosopt(temp[time_var].to_numpy(), temp[predict_var].to_numpy())
        phases.append(temp_phase)
        temp = pd.concat([temp.reset_index().drop(columns=['index']), pd.DataFrame(temp_X_fit, columns=columns)], axis=1)
        if type(df_new)==int:
            df_new=temp.copy()
        else:
            df_new = pd.concat([df_new, temp], axis=0,ignore_index=True)

    phase=np.array(phases).mean()
    X_test = np.linspace(0, 100, 1000)
    X_fit_test = np.hstack((0 * X_test[:, None] + 1, np.cos(X_test[:, None] * 2 * np.pi / period + phase)))

    if interactions==None and variables==None:
        formula = predict_var + "~"
        for i,col in enumerate(columns):
            if i==0:
                formula=formula+col
            else:
                formula=formula+"+"+col
        formula="Y~rr1"
        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        if summary:
            print(mdf.summary())

        temp = pd.DataFrame(X_fit_test, columns=columns)
        temp_vals = mdf.predict(temp)
        temp['predicted'] = temp_vals
        rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
        rhythm_params['Y_test'] = temp_vals
        df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],ignore_index=True)
        if plot:
            plotter.subplot_model(X_test, temp['predicted'], X_test, temp['predicted'], ax, plot_model=True,
                                  plot_measurements=False, color=colors[0])
            plotter.subplot_model(df_new[time_var], df_new[predict_var],
                                  df_new[time_var], df_new[predict_var], ax, plot_model=False,
                                  plot_measurements=True, color=colors[0],
                                  plot_measurements_with_color=True)


    elif interactions!=None:
        for var in interactions:
            df_new = df_new[~df_new[var].isna()]
        formula= predict_var + helpers.formulate_formula_interaction(interactions, columns)

        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        if summary:
            print(mdf.summary())

        combos = []
        for var in interactions:
            combos.append(df_new[var].unique())

        combos = list(itertools.product(*combos))
        for i, combo in enumerate(combos):
            temp = pd.DataFrame(X_fit_test, columns=columns)
            for ix in range(len(combo)):
                temp[interactions[ix]] = combo[ix]
            temp_vals = mdf.predict(temp)
            temp['predicted'] = temp_vals
            rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
            rhythm_params['parameter'] = combo
            df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],
                                   ignore_index=True)
            if plot:
                plotter.subplot_model(X_test, temp['predicted'], X_test, temp['predicted'], ax, plot_model=True,
                                      plot_measurements=False, fit_label=str(combo), color=colors[i])
    elif variables!=None:
        for var in variables:
           df_new = df_new[~df_new[var].isna()]
        formula = predict_var + helpers.formulate_formula(variables, columns)

        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        if summary:
            print(mdf.summary())

        combos = []
        for var in variables:
           combos.append(df_new[var].unique())

        temp = pd.DataFrame(X_fit_test, columns=columns)
        combos = list(itertools.product(*combos))
        for i, combo in enumerate(combos):
            temp = pd.DataFrame(X_fit_test, columns=columns)
            measurements_df=df_new.copy()
            for ix in range(len(combo)):
                temp[variables[ix]] = combo[ix]
                measurements_df=measurements_df[measurements_df[variables[ix]]==combo[ix]]
            temp_vals = mdf.predict(temp)
            temp['predicted'] = temp_vals
            rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
            rhythm_params['parameter'] = combo
            df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],
                                       ignore_index=True)
            if plot:
                plotter.subplot_model(X_test, temp['predicted'], X_test, temp['predicted'], ax, plot_model=True,
                                      plot_measurements=False, fit_label=str(combo), color=colors[i])
                plotter.subplot_model(measurements_df[time_var], measurements_df[predict_var],
                                      measurements_df[time_var], measurements_df[predict_var], ax, plot_model=False,
                                      plot_measurements=True, fit_label=str(combo), color=colors[i],
                                      plot_measurements_with_color=True)

    if plot:
        plt.title(plot_title)
        ax.legend()
        fig.savefig(save_to)
        plt.show()
    return df_results

def me_arser(df, predict_var, time_var, group_var, n_periods=1,variables=None,interactions=None,random_var="1",plot=True,est_periods=-1, save_to='',plot_title='',summary=False):
    df_results=pd.DataFrame()
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    df=df.reset_index(drop=True)
    columns= helpers.get_column_names(n_periods)

    df_new = -1
    periods=[]
    for id in df[group_var].unique():
        temp=df[df[group_var]==id].copy()

        a = temp['data_params'].iloc[0]
        b = ast.literal_eval(a)
        time_step = b['step']

        if(type(est_periods)==int):
            temp_X_fit, temp_X_test, temp_X_fit_test, temp_X_fit_eval_params, temp_periods= methods.arser_eval_gee(temp[time_var].to_numpy(), temp[predict_var].to_numpy(), time_step=time_step, n_periods=n_periods)
            temp_X_fit = temp_X_fit[:, 1::]

        else:
            temp_est_periods=est_periods[est_periods.id==id]
            temp_est_periods = temp_est_periods[temp_est_periods.n_periods == n_periods]
            temp_est_periods=temp_est_periods.iloc[0]['periods']
            temp_est_periods = temp_est_periods.split('[')[1].split(']')[0]
            temp_periods = np.fromstring(temp_est_periods, sep=',')
            temp_X_fit, temp_X_test, temp_X_fit_test, temp_X_fit_eval_params = methods.arser(temp[time_var].to_numpy(), temp_periods)

        temp = pd.concat([temp.reset_index().drop(columns=['index']), pd.DataFrame(temp_X_fit, columns=columns)], axis=1)

        if type(df_new)==int:
            df_new=temp.copy()
            periods = temp_periods.copy()
        else:
            df_new = pd.concat([df_new, temp], axis=0,ignore_index=True)
            periods = np.vstack([periods, temp_periods])

    mean_periods = np.mean(periods, axis=0)
    if len(periods)==1:
        mean_periods = [mean_periods]

    X_test = np.linspace(0, 100, 1000)
    X_fit_test = np.array([])
    X_fit_test = X_fit_test.reshape(len(X_test), 0)
    for T in mean_periods:
        B_test = np.cos((X_test / (T)) * np.pi * 2)
        A_test = np.sin((X_test / (T)) * np.pi * 2)
        X_fit_test = np.column_stack((X_fit_test, A_test, B_test))

    if interactions==None and variables==None:
        formula = predict_var + "~"
        for i,col in enumerate(columns):
            if i==0:
                formula=formula+col
            else:
                formula=formula+"+"+col

        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        if summary:
            print(mdf.summary())

        temp = pd.DataFrame(X_fit_test, columns=columns)
        temp_vals = mdf.predict(temp)
        temp['predicted'] = temp_vals
        rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
        rhythm_params['Y_test']=temp_vals
        df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],ignore_index=True)
        if plot:
            plotter.subplot_model(X_test, temp['predicted'], X_test, temp['predicted'], ax, plot_model=True,
                                  plot_measurements=False, color=colors[0])
            plotter.subplot_model(df_new[time_var], df_new[predict_var],
                                  df_new[time_var], df_new[predict_var], ax, plot_model=False,
                                  plot_measurements=True, color=colors[0],
                                  plot_measurements_with_color=True)


    elif interactions!=None:
        for var in interactions:
            df_new = df_new[~df_new[var].isna()]
        formula= predict_var + helpers.formulate_formula_interaction(interactions, columns)

        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        if summary:
            print(mdf.summary())

        combos = []
        for var in interactions:
            combos.append(df_new[var].unique())

        combos = list(itertools.product(*combos))
        for i, combo in enumerate(combos):
            temp = pd.DataFrame(X_fit_test, columns=columns)
            for ix in range(len(combo)):
                temp[interactions[ix]] = combo[ix]
            temp_vals = mdf.predict(temp)
            temp['predicted'] = temp_vals
            rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
            rhythm_params['parameter'] = combo
            df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],
                                   ignore_index=True)
            if plot:
                plotter.subplot_model(X_test, temp['predicted'], X_test, temp['predicted'], ax, plot_model=True,
                                      plot_measurements=False, fit_label=str(combo), color=colors[i])
    elif variables!=None:
        for var in variables:
           df_new = df_new[~df_new[var].isna()]
        formula = predict_var + helpers.formulate_formula(variables, columns)

        md = smf.mixedlm(formula, df_new, groups=df_new[group_var], re_formula=random_var)
        mdf = md.fit()

        if summary:
            print(mdf.summary())

        combos = []
        for var in variables:
           combos.append(df_new[var].unique())

        temp = pd.DataFrame(X_fit_test, columns=columns)
        combos = list(itertools.product(*combos))
        for i, combo in enumerate(combos):
            temp = pd.DataFrame(X_fit_test, columns=columns)
            measurements_df=df_new.copy()
            for ix in range(len(combo)):
                temp[variables[ix]] = combo[ix]
                measurements_df=measurements_df[measurements_df[variables[ix]]==combo[ix]]
            temp_vals = mdf.predict(temp)
            temp['predicted'] = temp_vals
            rhythm_params = dproc.evaluate_rhythm_params(X_test, temp_vals)
            rhythm_params['parameter'] = combo
            df_results = pd.concat([df_results, (pd.DataFrame.from_dict(rhythm_params, orient='index')).T],
                                       ignore_index=True)
            if plot:
                plotter.subplot_model(X_test, temp['predicted'], X_test, temp['predicted'], ax, plot_model=True,
                                      plot_measurements=False, fit_label=str(combo), color=colors[i])
                plotter.subplot_model(measurements_df[time_var], measurements_df[predict_var],
                                      measurements_df[time_var], measurements_df[predict_var], ax, plot_model=False,
                                      plot_measurements=True, fit_label=str(combo), color=colors[i],
                                      plot_measurements_with_color=True)

    if plot:
        plt.title(plot_title)
        ax.legend()
        fig.savefig(save_to)
        plt.show()
    return df_results
