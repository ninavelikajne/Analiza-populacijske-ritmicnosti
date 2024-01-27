import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import circstd, circmean
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from fit import helpers, methods, plotter
from fit.helpers import project_acr, phase_to_radians, amp_acr, acrophase_to_hours


def fit_to_model(df, method, n_components=[1, 2, 3], period=[24], phase=np.linspace(-np.pi, np.pi, 100)):
    X = df['X'].to_numpy()
    Y = df['Y'].to_numpy()

    if method == 'cosinor':
        df_result = methods.cosinor_eval(X, Y, n_components=n_components, period=period[0])
    elif method == 'cosopt':
        df_result = methods.cosopt_eval(X, Y, period=period[0], phase=phase)
    elif method == "arser":
        df_result = methods.arser_eval(X, Y)

    return df_result


def evaluate_cosinor1(x, MESOR, amp, acr, period):
    return MESOR + amp * np.cos((2 * np.pi * x / period) + acr)


def non_population_fit_cosinor1(df, period=24, corrected=True):
    rrr = np.cos(2 * np.pi * df.X / period)
    sss = np.sin(2 * np.pi * df.X / period)

    df['rrr'] = rrr
    df['sss'] = sss

    results = smf.ols('Y ~ rrr + sss', df).fit()

    beta_s = results.params['sss']
    beta_r = results.params['rrr']
    amp, acr = amp_acr(beta_s, beta_r)
    # project acropahse to interval -pi,pi
    acr = helpers.project_acr(acr)

    vmat = results.cov_params().loc[['rrr', 'sss'], ['rrr', 'sss']]
    indVmat = vmat

    a_r = (beta_r ** 2 + beta_s ** 2) ** (-0.5) * beta_r
    a_s = (beta_r ** 2 + beta_s ** 2) ** (-0.5) * beta_s
    b_r = (1 / (1 + (beta_s ** 2 / beta_r ** 2))) * (-beta_s / beta_r ** 2)
    b_s = (1 / (1 + (beta_s ** 2 / beta_r ** 2))) * (1 / beta_r)

    if corrected:
        b_r = -b_r
        b_s = -b_s

    jac = np.array([[a_r, a_s], [b_r, b_s]])

    cov_trans = np.dot(np.dot(jac, indVmat), np.transpose(jac))
    se_trans_only = np.sqrt(np.diag(cov_trans))
    zt = abs(stats.norm.ppf((1 - 0.95) / 2))

    trans_names = [results.params.index.values[0]] + ['amp', 'acr']

    coef_trans = np.array([results.params.iloc[0], amp, acr])
    se_trans = np.concatenate((np.sqrt(np.diag(results.cov_params().loc[['Intercept'], ['Intercept']])), se_trans_only))

    lower_CI_trans = coef_trans - np.abs(zt * se_trans)
    upper_CI_trans = coef_trans + np.abs(zt * se_trans)
    p_value_trans = 2 * stats.norm.cdf(-np.abs(coef_trans / se_trans))

    statistics = {'parameters': trans_names,
                  'values': coef_trans,
                  'SE': se_trans,
                  'CI': (lower_CI_trans, upper_CI_trans),
                  'p-values': p_value_trans,
                  'F-test': results.f_pvalue}

    return results, amp, acr, statistics


def population_fit_cosinor1(df_pop, period, save_to='', alpha=0.05, plot_on=True, plot_individuals=True,
                            plot_measurements=True, plot_margins=True, color="black", hold_on=False,
                            plot_residuals=False, save_folder=''):
    population_id = df_pop.population_id.iloc[0]

    if save_folder:
        # save_to = save_folder+"\\pop_"+name+'.pdf'
        save_to = os.path.join(save_folder, "pop_" + population_id)
    else:
        save_to = ""
    params = -1
    tests = df_pop.id.unique()
    k = len(tests)
    param_names = ['Intercept', 'rrr (beta)', 'sss (gamma)', 'amp', 'acr']
    cosinors = []
    df_pop = df_pop.rename(columns={"X": "x", "Y": "y"})
    # test_name = tests[0].split('_rep')[0]

    min_X = np.min(df_pop.x.values)
    max_X = np.max(df_pop.x.values)
    X_fit = np.linspace(0, 23, 100)

    for test in tests:
        x, y = df_pop[df_pop.id == test].x.values, df_pop[df_pop.id == test].y.values
        fit_results, amp, acr, _ = methods.fit_cosinor(x, y, period=period, save_to=save_to, plot_on=False)
        if plot_on and plot_individuals:
            # X_fit = np.linspace(min(x), max(x), 100)
            rrr_fit = np.cos(2 * np.pi * X_fit / period)
            sss_fit = np.sin(2 * np.pi * X_fit / period)

            data = pd.DataFrame()
            data['rrr'] = rrr_fit
            data['sss'] = sss_fit
            Y_fit = fit_results.predict(data).values

            plt.plot(X_fit, Y_fit, color=color, alpha=0.1, label='_Hidden label')

            if plot_residuals:
                plt.figure(2)

                resid = fit_results.resid
                sm.qqplot(resid)
                plt.title(test)
                if save_to:
                    plt.savefig(save_to + f'_resid_{test}' + '.pdf')
                    plt.savefig(save_to + f'_resid_{test}' + '.png')
                    plt.close()
                else:
                    plt.show()
                plt.figure(1)

            # M = fit_results.params[0]
            # y_fit = evaluate_cosinor(x, M, amp, acr, period)
            # plt.plot(x, y_fit, 'k')
        if plot_on and plot_measurements:
            plt.plot(x, y, 'o', color=color, markersize=1, label='_Hidden label')

        if type(params) == int:
            params = np.append(fit_results.params, np.array([amp, acr]))
            if plot_on and plot_margins:
                Y_fit_all = Y_fit
        else:
            params = np.vstack([params, np.append(fit_results.params, np.array([amp, acr]))])
            if plot_on and plot_margins:
                Y_fit_all = np.vstack([Y_fit_all, Y_fit])

        cosinors.append(fit_results)

    if k > 1:
        means = np.mean(params, axis=0)
    else:
        means = params
    MESOR = means[0]
    beta = means[1]
    gamma = means[2]

    amp, acr = amp_acr(gamma, beta)
    means[3] = amp
    means[4] = acr

    if k > 1:
        sd = np.std(params, axis=0, ddof=1)
        sdm = sd[0]
        sdb = sd[1]
        sdy = sd[2]

        covby = np.cov(params[:, 1], params[:, 2])[0, 1]
        denom = (amp ** 2) * k
        c22 = (((sdb ** 2) * (beta ** 2)) + (2 * covby * beta * gamma) + ((sdy ** 2) * (gamma ** 2))) / denom
        c23 = (((-1 * ((sdb ** 2) - (sdy ** 2))) * (beta * gamma)) + (covby * ((beta ** 2) - (gamma ** 2)))) / denom
        c33 = (((sdb ** 2) * (gamma ** 2)) - (2 * covby * beta * gamma) + ((sdy ** 2) * (beta ** 2))) / denom

        t = abs(stats.t.ppf(alpha / 2, df=k - 1))

        mesoru = MESOR + ((t * sdm) / (k ** 0.5))
        mesorl = MESOR - ((t * sdm) / (k ** 0.5))

        sem = sdm / (k ** 0.5)
        T0 = MESOR / sem
        p_mesor = 2 * (1 - stats.t.cdf(abs(T0), k - 1))
        # p_mesor = 2 * stats.norm.cdf(-np.abs(MESOR/sem))

        ampu = amp + (t * (c22 ** 0.5))
        ampl = amp - (t * (c22 ** 0.5))
        se_amp = c22 ** 0.5

        T0 = amp / se_amp
        p_amp = 2 * (1 - stats.t.cdf(abs(T0), k - 1))
        # p_amp = 2 * stats.norm.cdf(-np.abs(amp/se_amp))

        if (ampu > 0 and ampl < 0):
            fiu = np.nan
            fil = np.nan
            p_acr = 1
            print(
                "Warning: Amplitude confidence interval contains zero. Acrophase confidence interval cannot be calculated and was set to NA.")
        else:
            fiu = acr + np.arctan(((c23 * (t ** 2)) + (
                    (t * np.sqrt(c33)) * np.sqrt((amp ** 2) - (((c22 * c33) - (c23 ** 2)) * ((t ** 2) / c33))))) / (
                                          (amp ** 2) - (c22 * (t ** 2))))
            fil = acr + np.arctan(((c23 * (t ** 2)) - (
                    (t * np.sqrt(c33)) * np.sqrt((amp ** 2) - (((c22 * c33) - (c23 ** 2)) * ((t ** 2) / c33))))) / (
                                          (amp ** 2) - (c22 * (t ** 2))))

            se_acr = (fiu - acr) / t
            T0 = acr / se_acr
            p_acr = 2 * (1 - stats.t.cdf(abs(T0), k - 1))
    else:
        mesoru = MESOR
        mesorl = MESOR
        ampu = amp
        ampl = amp
        fiu = acr
        fil = acr

        p_acr = 1
        p_amp = 1
        p_mesor = 1

    Y_fit = evaluate_cosinor1(X_fit, MESOR, amp, acr, period)
    if plot_on:
        plt.plot(X_fit, Y_fit, color=color, label=str(test))
        plt.title('pop_id')

        if plot_margins:
            """
            me = np.linspace(mesoru, mesorl, 5)
            am = np.linspace(ampu, ampl, 5)
            fi = np.linspace(fiu, fil, 5)

            Y = y

            for m in me:
                for a in am:
                    for f in fi:
                        yy = evaluate_cosinor(x, m, a, f, period)
                        Y = np.vstack([Y, yy])
                        #plt.plot(x,yy)


            lower = np.min(Y, axis=0)        
            upper = np.max(Y, axis=0)        

            plt.fill_between(x, lower, upper, color='black', alpha=0.1)  
            """
            if k <= 1:
                _, lower, upper = wls_prediction_std(fit_results, exog=sm.add_constant(data, has_constant='add'),
                                                     alpha=0.05)
            else:
                var_Y = np.var(Y_fit_all, axis=0, ddof=k - 1)
                sd_Y = var_Y ** 0.5
                lower = Y_fit - ((t * sd_Y) / (k ** 0.5))  # biased se as above
                upper = Y_fit + ((t * sd_Y) / (k ** 0.5))  # biased se as above
            plt.fill_between(X_fit, lower, upper, color=color, alpha=0.1)

        if not hold_on:
            if save_to:
                plt.savefig(save_to + '.pdf')
                plt.savefig(save_to + '.png')
                plt.close()
            else:
                plt.show()

    confint = {'amp': (ampl, ampu),
               'acr': (fil, fiu),
               'MESOR': (mesorl, mesoru)}

    # calculate the overall p-value
    if k > 1:
        betas = params[:, 1]
        gammas = params[:, 2]
        r = np.corrcoef(betas, gammas)[0, 1]
        frac1 = (k * (k - 2)) / (2 * (k - 1))
        frac2 = 1 / (1 - r ** 2)
        frac3 = beta ** 2 / sdb ** 2
        frac4 = (beta * gamma) / (sdb * sdy)
        frac5 = gamma ** 2 / sdy ** 2
        brack = frac3 - (2 * r * frac4) + frac5
        Fvalue = frac1 * frac2 * brack
        df2 = k - 2
        p_value = 1 - stats.f.cdf(Fvalue, 2, df2)

    else:
        p_value = np.nan

    res = {'population_id': population_id, 'names': param_names, 'values': params, 'means': means, 'confint': confint,
           'p_value': p_value, 'p_mesor': p_mesor, 'p_amp': p_amp, 'p_acr': p_acr}

    d = {'population_id': population_id,
         'p': res['p_value'],
         'amplitude': res['means'][-2],
         'p(amplitude)': res['p_amp'],
         'CI(amplitude)': [res['confint']['amp'][0], res['confint']['amp'][1]],
         'p(mesor)': res['p_mesor'],
         'mesor': res['means'][0],
         'CI(mesor)': [res['confint']['MESOR'][0], res['confint']['MESOR'][1]],
         'acrophase': res['means'][-1],
         'p(acrophase)': res['p_acr'],
         'CI(acrophase)': [res['confint']['acr'][0], res['confint']['acr'][1]],
         'acrophase[h]': acrophase_to_hours(res['means'][-1], period),
         'Y_test': Y_fit}

    df_result = (pd.DataFrame.from_dict(d, orient='index')).T

    return df_result


def fit_non_population(df, method, n_components=1, n_periods=1, period=[24], phase=np.linspace(-np.pi, 0, 100),
                       plot=True, ax=None, time_step=1, fig_name='', label='', color='gray', raw_label=''):
    X = df['X'].to_numpy()
    Y = df['Y'].to_numpy()

    if plot and ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    if plot:
        if raw_label != '':
            plotter.subplot_model(X, Y, X, Y, ax, plot_model=False, raw_label=raw_label,
                                  plot_measurements_with_color=True, color=color)
        else:
            plotter.subplot_model(X, Y, X, Y, ax, plot_model=False)

    if method == 'cosinor':
        result, model = methods.cosinor_eval(X, Y, n_components=n_components, period=period[0])
    elif method == 'cosopt':
        result, model = methods.cosopt_eval(X, Y, period=period[0], phase=phase)
    elif method == "arser":
        result, model = methods.arser_eval(X, Y, n_periods=n_periods, time_step=time_step)

    if type(result) != int:
        result = result.iloc[0]
        if plot:
            if label != '':
                plotter.subplot_model(X, Y, result['X_test'], result['Y_test'], ax,
                                      fit_label=label, plot_measurements=False, color=color)
            else:
                plotter.subplot_model(X, Y, result['X_test'], result['Y_test'], ax,
                                      fit_label=method, plot_measurements=False, color=color)

            if fig_name != '':
                plt.savefig('./results/figs_gen/' + fig_name + '_population_' + method + ".png")
                plt.show()
                plt.close()

    return result, ax


def fit_population(df, method, n_components=2, n_periods=2, period=[24], phase=np.linspace(-np.pi, 0, 100),
                   plot_individuals=True, plot_population=True, time_step=1, fig_name='', df_periods=None, pop_id=0):
    parameters_to_analyse = ['amplitude', 'acrophase', 'mesor']
    parameters_angular = ['acrophase']
    ind_params = {}
    ind_params_stats = {}
    for param in parameters_to_analyse:
        ind_params[param] = []

    plot = plot_individuals | plot_population
    X = df['X'].to_numpy()
    Y = df['Y'].to_numpy()

    params = -1
    replicates = df['id'].unique()
    k = len(replicates)
    err_cnt = 0

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        plotter.subplot_model(X, Y, X, Y, ax, plot_model=False)

    # individuals
    periods = []
    for replicate in replicates:
        rep = df[df.id == replicate]
        rep_x = rep['X'].to_numpy()
        rep_y = rep['Y'].to_numpy()

        if method == 'cosinor':
            rep_result, model = methods.cosinor_eval(rep_x, rep_y, n_components=n_components, period=period[0])
            rep_result['replicate'] = replicate
            rep_result = rep_result.iloc[0]
        elif method == 'cosopt':
            rep_result, model = methods.cosopt_eval(rep_x, rep_y, period=period[0], phase=phase)
            rep_result['replicate'] = replicate
            rep_result = rep_result.iloc[0]
        elif method == "arser":
            rep_result, model = methods.arser_eval(rep_x, rep_y, n_periods=n_periods, time_step=time_step)
            if type(rep_result) == int:
                err_cnt = err_cnt + 1
                continue
            rep_result['replicate'] = replicate
            rep_result = rep_result.iloc[0]

            new_row = {'n_periods': n_periods, 'population_id': pop_id, 'id': replicate,
                       'periods': rep_result['method_params']['period']}
            df_periods = pd.concat([df_periods, (pd.DataFrame.from_dict(new_row, orient='index')).T],
                                   ignore_index=True)

            if len(periods) == 0:
                periods = rep_result['method_params']['period']
            else:
                periods = np.vstack([periods, rep_result['method_params']['period']])

        for param in parameters_to_analyse:
            ind_params[param].append(rep_result[param])

        results = rep_result['results']
        rep_params = results.params
        if type(params) == int:
            params = rep_params
            Y_test_all = rep_result['Y_test']
        else:
            params = np.vstack([params, rep_params])
            Y_test_all = np.vstack([Y_test_all, rep_result['Y_test']])

        if plot_individuals:
            plotter.subplot_model(X, Y, rep_result['X_test'], rep_result['Y_test'], ax,
                                  fit_label=replicate, plot_measurements=False, color='gray')
    k = k - err_cnt
    if k == 0:
        return -1
    if k > 1:
        means = np.mean(params, axis=0)
        variances = np.sum((params - np.mean(params, axis=0)) ** 2, axis=0) / (
                k - 1)  # np.var(params, axis=0) # isto kot var z ddof=k-1
        sd = variances ** 0.5
        se = sd / ((k - 1) ** 0.5)
        T0 = means / se
        p_values = 2 * (1 - stats.t.cdf(abs(T0), k - 1))
        t = abs(stats.t.ppf(0.05 / 2, df=k - 1))
        lower_CI = means - ((t * sd) / ((k - 1) ** 0.5))
        upper_CI = means + ((t * sd) / ((k - 1) ** 0.5))
        results.initialize(model, means)

        # analysis of amplitude, mesor, acrophase...
        for ind_param in parameters_to_analyse:
            ind_vals = np.array(ind_params[ind_param])

            if ind_param in parameters_angular:
                p_means = project_acr(circmean(ind_vals, high=np.pi, low=-np.pi))
                p_sd = circstd(ind_vals, high=np.pi, low=-np.pi)
            else:
                p_means = np.mean(ind_vals)
                p_variances = np.sum((ind_vals - np.mean(ind_vals)) ** 2) / (k - 1)
                p_sd = p_variances ** 0.5

            p_se = p_sd / ((k - 1) ** 0.5)
            p_T0 = p_means / p_se
            ind_param_p = 2 * (1 - stats.t.cdf(abs(p_T0), k - 1))
            t = abs(stats.t.ppf(0.05 / 2, df=k - 1))
            ind_param_lower_CI = p_means - ((t * p_sd) / ((k - 1) ** 0.5))
            ind_param_upper_CI = p_means + ((t * p_sd) / ((k - 1) ** 0.5))

            ind_params_stats[f'mean({ind_param})'] = p_means
            ind_params_stats[f'p({ind_param})'] = ind_param_p
            ind_params_stats[f'CI({ind_param})'] = [ind_param_lower_CI, ind_param_upper_CI]
    else:
        means = params
        sd = np.zeros(len(params))
        sd[:] = np.nan
        se = np.zeros(len(params))
        se[:] = np.nan
        lower_CI = means
        upper_CI = means
        p_values = np.zeros(len(params))
        p_values[:] = np.nan

        for ind_param in parameters_to_analyse:
            ind_params_stats[f'mean({ind_param})'] = ind_params[ind_param][0]
            ind_params_stats[f'p({ind_param})'] = np.nan
            ind_params_stats[f'CI({ind_param})'] = [np.nan, np.nan_to_num]

    xy = list(zip(X, Y))
    xy.sort()
    x, y = zip(*xy)
    x, y = np.array(x), np.array(y)
    if method == 'cosinor':
        X_fit, X_test, X_fit_test, X_fit_eval_params = methods.cosinor(x, n_components=n_components, period=period[0])
        method_params = {'n_components': n_components, 'period': period[0]}

        X_fit = sm.add_constant(X_fit, has_constant='add')
        X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
        X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')
    elif method == 'cosopt':
        X_fit, X_test, X_fit_test, X_fit_eval_params, best_phase = methods.cosopt(x, y, period=period[0], phase=phase)
        method_params = {'phase': best_phase, 'period': period[0]}
    elif method == 'arser':
        # ex=np.rint(periods)
        # modus=stats.mode(ex,axis=0).mode[0]
        mean_periods = np.mean(periods, axis=0)
        if k == 1:
            mean_periods = [mean_periods]
        X_fit, X_test, X_fit_test, X_fit_eval_params = methods.arser(x, period=mean_periods)
        method_params = {'period': mean_periods, 'n_periods': n_periods}

        X_fit = sm.add_constant(X_fit, has_constant='add')
        X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
        X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')

    Y_test = results.predict(X_fit_test)
    Y_eval_params = results.predict(X_fit_eval_params)
    Y_fit = results.predict(X_fit)

    # CI plot
    var_Y = np.var(Y_test_all, axis=0, ddof=k - 1)
    sd_Y = var_Y ** 0.5
    # lowerA = Y_test - ((t * sd_Y) / ((k - 1) ** 0.5))
    # upperA = Y_test + ((t * sd_Y) / ((k - 1) ** 0.5))

    rhythm_params = evaluate_rhythm_params(X_test, Y_eval_params)
    df_result = calculate_statistics(Y, Y_fit, results, method, method_params, rhythm_params)
    statistics_params = {'values': means, 'SE': se, 'CI': (lower_CI, upper_CI), 'p-values': p_values,
                         'ind_params_stats': ind_params_stats}

    df_result.update({'data_mean': np.mean(Y)})
    df_result.update({'data_std': np.std(Y)})
    df_result.update({'params': params})
    df_result.update({'statistics_params': statistics_params})
    df_result.update({'results': results})
    df_result.update({'X_test': X_test})
    df_result.update({'Y_test': Y_test})
    df_result.update({'X_fit_test': X_fit_test})

    df_result = (pd.DataFrame.from_dict(df_result, orient='index')).T

    if plot_population:
        plotter.subplot_model(X, Y, X_test, Y_test, ax,
                              fit_label='population',
                              plot_measurements=False, color='blue')
    if plot:
        plt.savefig('./results/figs_gen/' + fig_name + '_population_' + method + ".png")
        # plt.show()
        plt.close()

    return df_result, df_periods


def evaluate_rhythm_params(X, Y, period=24):
    X = X[:period * 10]
    Y = Y[:period * 10]
    m = min(Y)
    M = max(Y)
    A = M - m
    MESOR = m + A / 2
    AMPLITUDE = abs(A / 2)
    PHASE = 0
    PHASE_LOC = 0

    locs, heights = signal.find_peaks(Y, height=M * 0.75)

    if len(locs) >= 1:
        PHASE = X[locs[0]]
        PHASE_LOC = locs[0]

    if period:
        ACROPHASE = phase_to_radians(PHASE, period)
        # ACROPHASE = project_acr(ACROPHASE)
    else:
        ACROPHASE = np.nan

    heights = heights['peak_heights']
    x = np.take(X, locs)

    result = {'amplitude': round(AMPLITUDE, 2), 'acrophase': ACROPHASE, 'mesor': round(MESOR, 2),
              'locs': np.around(x, decimals=2),
              'heights': np.around(heights, decimals=2)}
    return result


def calculate_statistics(Y, Y_fit, results, method, method_params, rhythm_param):
    # RSS
    RSS = sum((Y - Y_fit) ** 2)

    # AIC
    aic = results.aic

    # BIC
    bic = results.bic

    # resid
    resid = results.resid
    resid_mean = np.mean(resid)
    resid_std = np.std(resid)

    return {'method': method, 'method_params': method_params,
            'amplitude': rhythm_param['amplitude'], 'acrophase': rhythm_param['acrophase'],
            'mesor': rhythm_param['mesor'], 'peaks': rhythm_param['locs'], 'heights': rhythm_param['heights'],
            'RSS': RSS, 'AIC': aic, 'BIC': bic,
            'log_likelihood': results.llf,
            'resid': resid, 'resid_mean': resid_mean, 'resid_std': resid_std,
            'est_mean': Y_fit.mean(), 'est_std': Y_fit.std(), 'Y_est': Y_fit}
