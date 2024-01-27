import sys

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from fit import dproc
from fit import helpers as hlp
from fit import plotter

r = robjects.r
spec_ar = r['spec.ar']


def cosinor_single(data, period=24, corrected=True):
    rrr = np.cos(2 * np.pi * data.x / period)
    sss = np.sin(2 * np.pi * data.x / period)

    data['rrr'] = rrr
    data['sss'] = sss

    results = smf.ols('y ~ rrr + sss', data).fit()

    beta_s = results.params['sss']
    beta_r = results.params['rrr']
    amp, acr = hlp.amp_acr(beta_s, beta_r)
    # project acropahse to interval -pi,pi
    acr = hlp.project_acr(acr)

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


def fit_cosinor(X, Y, period, test='', save_to='', plot_on=True):
    data = pd.DataFrame()
    data['x'] = X
    data['y'] = Y

    # fit_results, amp, acr = fit_cosinor_df(data, period)

    fit_results, amp, acr, statistics = cosinor_single(data, period)
    if plot_on:
        plotter.plot_single(data, fit_results, test=test, plot_measurements=True, save_to=save_to, plot_dense=True,
                            period=period)

    return fit_results, amp, acr, statistics


def cosinor(X, n_components, period=24):
    X_test = np.linspace(0, 100, 1000)

    for i in range(n_components):
        k = i + 1
        A = np.sin((X / (period / k)) * np.pi * 2)
        B = np.cos((X / (period / k)) * np.pi * 2)

        A_test = np.sin((X_test / (period / k)) * np.pi * 2)
        B_test = np.cos((X_test / (period / k)) * np.pi * 2)

        if i == 0:
            X_fit = np.column_stack((A, B))
            X_fit_test = np.column_stack((A_test, B_test))
        else:
            X_fit = np.column_stack((X_fit, A, B))
            X_fit_test = np.column_stack((X_fit_test, A_test, B_test))

    X_fit_eval_params = X_fit_test

    return X_fit, X_test, X_fit_test, X_fit_eval_params


def cosinor_eval(X, Y, n_components=2, period=24):
    X_fit, X_test, X_fit_test, X_fit_eval_params = cosinor(X, n_components=n_components, period=period)
    method_params = {'n_components': n_components, 'period': period}

    X_fit = sm.add_constant(X_fit, has_constant='add')
    X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
    X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')

    model = sm.OLS(Y, X_fit)
    results = model.fit()

    Y_test = results.predict(X_fit_test)
    Y_eval_params = results.predict(X_fit_eval_params)
    Y_fit = results.predict(X_fit)

    rhythm_params = dproc.evaluate_rhythm_params(X_test, Y_eval_params)
    df_result = dproc.calculate_statistics(Y, Y_fit, results, 'cosinor', method_params, rhythm_params)
    df_result.update({'data_mean': np.mean(Y)})
    df_result.update({'data_std': np.std(Y)})
    df_result.update({'X_test': X_test})
    df_result.update({'Y_test': Y_test})
    df_result.update({'results': results})
    df_result.update({'X_fit_test': X_fit_test})

    df_result = (pd.DataFrame.from_dict(df_result, orient='index')).T

    return df_result, model


def fit_cosinor_components(X, Y, n_components=[1, 2, 3], period=24):
    df_results = pd.DataFrame()

    for component in n_components:
        X_fit, X_test, X_fit_test, X_fit_eval_params = cosinor(X, n_components=component, period=period)
        method_params = {'n_components': component, 'period': period}

        X_fit = sm.add_constant(X_fit, has_constant='add')
        X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
        X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')

        model = sm.OLS(Y, X_fit)
        results = model.fit()

        Y_test = results.predict(X_fit_test)
        Y_eval_params = results.predict(X_fit_eval_params)
        Y_fit = results.predict(X_fit)

        rhythm_params = dproc.evaluate_rhythm_params(X_test, Y_eval_params)
        df_result = dproc.calculate_statistics(Y, Y_fit, results, 'cosinor', method_params, rhythm_params)
        df_result.update({'data_mean': np.mean(Y)})
        df_result.update({'data_std': np.std(Y)})
        df_result.update({'X_test': X_test})
        df_result.update({'Y_test': Y_test})
        df_result.update({'results': results})
        df_result.update({'X_fit_test': X_fit_test})

    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(df_result, orient='index')).T], ignore_index=True)

    best_cosinor = hlp.get_best_n_components(df_results)
    best_cosinor = (best_cosinor.to_frame()).T
    return best_cosinor


def cosopt(X, Y, period=24, phase=np.linspace(-np.pi, 0, 100)):
    X = X[:, None]
    Y = Y[:, None]
    X_test = np.linspace(0, 100, 1000)
    MSE = np.zeros(len(phase))
    for i, pha in enumerate(phase):
        X_fit = np.hstack((np.ones(X.shape), np.cos(X * 2 * np.pi / period + pha)))
        try:
            C = np.dot(np.linalg.inv(np.dot(X_fit.T, X_fit)), np.dot(X_fit.T, Y))
            MSE[i] = np.mean((np.dot(X_fit, C) - Y) ** 2)
        except:
            continue

    i = np.unravel_index(MSE.argmin(), MSE.shape)
    X_fit = np.hstack((np.ones(X.shape), np.cos(X * 2 * np.pi / period + phase[i])))

    X_fit_test = np.hstack((0 * X_test[:, None] + 1, np.cos(X_test[:, None] * 2 * np.pi / period + phase[i])))
    X_fit_eval_params = X_fit_test

    return X_fit, X_test, X_fit_test, X_fit_eval_params, phase[i]


def cosopt_eval(X, Y, period=24, phase=np.linspace(-np.pi, np.pi, 100)):
    X_fit, X_test, X_fit_test, X_fit_eval_params, phase = cosopt(X, Y, period=period, phase=phase)
    method_params = {'phase': phase, 'period': period}

    # X_fit = sm.add_constant(X_fit, has_constant='add')
    # X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
    # X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')

    model = sm.OLS(Y, X_fit)
    results = model.fit()
    Y_test = results.predict(X_fit_test)
    Y_eval_params = results.predict(X_fit_eval_params)
    Y_fit = results.predict(X_fit)

    rhythm_params = dproc.evaluate_rhythm_params(X_test, Y_eval_params)
    df_result = dproc.calculate_statistics(Y, Y_fit, results, 'cosopt', method_params, rhythm_params)
    df_result.update({'data_mean': np.mean(Y)})
    df_result.update({'data_std': np.std(Y)})
    df_result.update({'X_test': X_test})
    df_result.update({'Y_test': Y_test})
    df_result.update({'results': results})
    df_result.update({'X_fit_test': X_fit_test})

    df_result = (pd.DataFrame.from_dict(df_result, orient='index')).T
    return df_result, model


def arser(X, period):
    X_fit = np.array([])
    X_fit = X_fit.reshape(len(X), 0)
    X_test = np.linspace(0, 100, 1000)
    X_fit_test = np.array([])
    X_fit_test = X_fit_test.reshape(len(X_test), 0)

    for T in period:
        A = np.sin((X / (T)) * np.pi * 2)
        B = np.cos((X / (T)) * np.pi * 2)
        X_fit = np.column_stack((X_fit, A, B))

        B_test = np.cos((X_test / (T)) * np.pi * 2)
        A_test = np.sin((X_test / (T)) * np.pi * 2)
        X_fit_test = np.column_stack((X_fit_test, A_test, B_test))

    X_fit_eval_params = X_fit_test
    return X_fit, X_test, X_fit_test, X_fit_eval_params


def arser_est_period(x, dt_y, is_filter=True, ar_method='mle', time_step=1):
    delta = time_step
    num_freq_mese = 500
    set_order = 24 / delta
    if (set_order == len(x)):
        set_order = len(x) / 2
    try:
        filter_y = hlp.savitzky_golay(dt_y)
    except:
        filter_y = hlp.savitzky_golay(dt_y, kernel=5, order=2)
    if is_filter:
        try:
            mese = spec_ar(robjects.FloatVector(filter_y.tolist()), n_freq=num_freq_mese, plot=False, method=ar_method,
                           order=set_order)
        except:
            print
            'spec_ar running error at line 70'
            sys.exit(1)
    else:
        try:
            mese = spec_ar(robjects.FloatVector(dt_y.tolist()), n_freq=num_freq_mese, plot=False, method=ar_method,
                           order=set_order)
        except:
            print
            'spec_ar running error at line 76'
            sys.exit(1)

    # search all the peaks of maximum entropy spectrum
    peaks_loc = []  # the location for each peak in mese spectrum
    for i in range(1, num_freq_mese - 1):
        if mese.rx2('spec')[i] > mese.rx2('spec')[i + 1] and mese.rx2('spec')[i] > mese.rx2('spec')[i - 1]:
            peaks_loc.append((mese.rx2('spec')[i], i))
    peaks_loc.sort(reverse=True)  # sort frequency by spectrum value
    try:
        periods = [1 / mese.rx2('freq')[item[1]] * delta for item in peaks_loc]
    except:
        periods = []
    return periods


def arser_eval_gee(X, Y, T_start=11, T_end=40, T_default=24, time_step=1, n_periods=2):
    offsetsL = [0, 1, 1]
    offsetsR = [0, 0, 1]
    is_filter = [False]
    ar_method = ['yule-walker', 'mle', 'burg']
    best_model = {'AIC': 1e6, 'period': None, 'filter': None, 'ar_method': ''}
    for p1 in is_filter:
        for p2 in ar_method:
            # choose best model's period from 'mle','yw','burg'
            try:
                est_periods = arser_est_period(X, Y, is_filter=p1, ar_method=p2, time_step=time_step)
            except:
                continue
            periods = list(filter((lambda x: x >= T_start and x <= T_end), est_periods))
            periods.sort(reverse=True)

            if (len(periods) == n_periods):
                periods = periods
            elif (len(periods) > n_periods):
                middle_index = (len(periods) // 2)
                offsetL = offsetsL[n_periods - 1]
                offsetR = offsetsR[n_periods - 1]
                periods = periods[(middle_index - offsetL): (middle_index + offsetR + 1)]
            elif (len(periods) < n_periods and len(periods) != 0):
                periods.sort()
                diff = n_periods - len(periods)
                steps = [5, 7, 12, 15, 18, 20]
                for ix in range(diff):
                    if (periods[len(periods) - 1] > (T_default / 2)):
                        periods.append(periods[len(periods) - 1] - steps[ix])
                    else:
                        periods.append(periods[len(periods) - 1] + steps[ix])
            else:
                steps = [0, 5, 7, 12, 15, 18, 20]
                for ix in range(n_periods):
                    periods.append(T_default - steps[ix])
            periods.sort()
            temp_X_fit, temp_X_test, temp_X_fit_test, temp_X_fit_eval_params = arser(X, periods)

            temp_X_fit = sm.add_constant(temp_X_fit, has_constant='add')
            temp_X_fit_test = sm.add_constant(temp_X_fit_test, has_constant='add')
            temp_X_fit_eval_params = sm.add_constant(temp_X_fit_eval_params, has_constant='add')

            model = sm.OLS(Y, temp_X_fit)
            results = model.fit()

            # period model selection by aic
            aic = results.aic
            if aic <= best_model['AIC']:
                best_model['AIC'] = aic
                best_model['period'] = periods
                best_model['filter'] = p1
                best_model['ar_method'] = p2
                best_model['results'] = results
                best_model['model'] = model
                X_fit = temp_X_fit
                X_test = temp_X_test
                X_fit_test = temp_X_fit_test
                X_fit_eval_params = temp_X_fit_eval_params

    try:
        results = best_model['results']
        method_params = {'period': best_model['period']}
        Y_test = results.predict(X_fit_test)
        Y_eval_params = results.predict(X_fit_eval_params)
        Y_fit = results.predict(X_fit)
        rhythm_params = dproc.evaluate_rhythm_params(X_test, Y_eval_params)
        df_result = dproc.calculate_statistics(Y, Y_fit, results, 'arser', method_params, rhythm_params)
        df_result.update({'data_mean': np.mean(Y)})
        df_result.update({'data_std': np.std(Y)})
        df_result.update({'X_test': X_test})
        df_result.update({'Y_test': Y_test})
        df_result.update({'results': results})
        df_result.update({'X_fit_test': X_fit_test})
        df_result = (pd.DataFrame.from_dict(df_result, orient='index')).T
    except:
        print("Can't estimate period")
        return -1, -1

    return X_fit, X_test, X_fit_test, X_fit_eval_params, best_model['period']


def arser_eval(X, Y, T_start=11, T_end=40, T_default=24, time_step=1, n_periods=2):
    offsetsL = [0, 1, 1]
    offsetsR = [0, 0, 1]
    is_filter = [False]
    ar_method = ['yule-walker', 'mle', 'burg']
    best_model = {'AIC': 1e6, 'period': None, 'filter': None, 'ar_method': ''}
    for p1 in is_filter:
        for p2 in ar_method:
            # choose best model's period from 'mle','yw','burg'
            try:
                est_periods = arser_est_period(X, Y, is_filter=p1, ar_method=p2, time_step=time_step)
            except:
                continue
            periods = list(filter((lambda x: x >= T_start and x <= T_end), est_periods))
            periods.sort(reverse=True)

            if (len(periods) == n_periods):
                periods = periods
            elif (len(periods) > n_periods):
                middle_index = (len(periods) // 2)
                offsetL = offsetsL[n_periods - 1]
                offsetR = offsetsR[n_periods - 1]
                periods = periods[(middle_index - offsetL): (middle_index + offsetR + 1)]
            elif (len(periods) < n_periods and len(periods) != 0):
                periods.sort()
                diff = n_periods - len(periods)
                steps = [5, 7, 12, 15, 18, 20]
                for ix in range(diff):
                    if (periods[len(periods) - 1] > (T_default / 2)):
                        periods.append(periods[len(periods) - 1] - steps[ix])
                    else:
                        periods.append(periods[len(periods) - 1] + steps[ix])
            else:
                steps = [0, 5, 7, 12, 15, 18, 20]
                for ix in range(n_periods):
                    periods.append(T_default - steps[ix])
            periods.sort()
            temp_X_fit, temp_X_test, temp_X_fit_test, temp_X_fit_eval_params = arser(X, periods)

            temp_X_fit = sm.add_constant(temp_X_fit, has_constant='add')
            temp_X_fit_test = sm.add_constant(temp_X_fit_test, has_constant='add')
            temp_X_fit_eval_params = sm.add_constant(temp_X_fit_eval_params, has_constant='add')

            model = sm.OLS(Y, temp_X_fit)
            results = model.fit()

            # period model selection by aic
            aic = results.aic
            if aic <= best_model['AIC']:
                best_model['AIC'] = aic
                best_model['period'] = periods
                best_model['filter'] = p1
                best_model['ar_method'] = p2
                best_model['results'] = results
                best_model['model'] = model
                X_fit = temp_X_fit
                X_test = temp_X_test
                X_fit_test = temp_X_fit_test
                X_fit_eval_params = temp_X_fit_eval_params

    try:
        results = best_model['results']
        method_params = {'period': best_model['period']}
        Y_test = results.predict(X_fit_test)
        Y_eval_params = results.predict(X_fit_eval_params)
        Y_fit = results.predict(X_fit)
        rhythm_params = dproc.evaluate_rhythm_params(X_test, Y_eval_params)
        df_result = dproc.calculate_statistics(Y, Y_fit, results, 'arser', method_params, rhythm_params)
        df_result.update({'data_mean': np.mean(Y)})
        df_result.update({'data_std': np.std(Y)})
        df_result.update({'X_test': X_test})
        df_result.update({'Y_test': Y_test})
        df_result.update({'results': results})
        df_result.update({'X_fit_test': X_fit_test})
        df_result = (pd.DataFrame.from_dict(df_result, orient='index')).T
    except:
        print("Can't estimate period")
        return -1, -1

    return df_result, best_model['model']


def arser2(X, Y, T_start=11, T_end=40, T_default=24, time_step=1, n_periods=1):
    offsetsL = [0, 1, 1]
    offsetsR = [0, 0, 1]
    is_filter = [False]
    ar_method = ['yule-walker', 'mle', 'burg']
    best_model = {'AIC': 1e6, 'period': None, 'filter': None, 'ar_method': ''}
    for p1 in is_filter:
        for p2 in ar_method:
            # choose best model's period from 'mle','yw','burg'
            try:
                est_periods = arser_est_period(X, Y, is_filter=p1, ar_method=p2, time_step=time_step)
            except:
                continue
            periods = list(filter((lambda x: x >= T_start and x <= T_end), est_periods))
            periods.sort(reverse=True)

            if (len(periods) == n_periods):
                periods = periods
            elif (len(periods) > n_periods):
                middle_index = (len(periods) // 2)
                offsetL = offsetsL[n_periods - 1]
                offsetR = offsetsR[n_periods - 1]
                periods = periods[(middle_index - offsetL): (middle_index + offsetR + 1)]
            elif (len(periods) < n_periods and len(periods) != 0):
                periods.sort()
                diff = n_periods - len(periods)
                steps = [5, 7, 12, 15, 18, 20]
                for ix in range(diff):
                    if (periods[len(periods) - 1] > (T_default / 2)):
                        periods.append(periods[len(periods) - 1] - steps[ix])
                    else:
                        periods.append(periods[len(periods) - 1] + steps[ix])
            else:
                steps = [0, 5, 7, 12, 15, 18, 20]
                for ix in range(n_periods):
                    periods.append(T_default - steps[ix])
            periods.sort()
            temp_X_fit, temp_X_test, temp_X_fit_test, temp_X_fit_eval_params = arser(X, periods)

            temp_X_fit = sm.add_constant(temp_X_fit, has_constant='add')
            temp_X_fit_test = sm.add_constant(temp_X_fit_test, has_constant='add')
            temp_X_fit_eval_params = sm.add_constant(temp_X_fit_eval_params, has_constant='add')

            model = sm.OLS(Y, temp_X_fit)
            results = model.fit()

            # period model selection by aic
            aic = results.aic
            if aic <= best_model['AIC']:
                best_model['AIC'] = aic
                best_model['period'] = periods
                best_model['filter'] = p1
                best_model['ar_method'] = p2
                best_model['results'] = results
                best_model['model'] = model
                X_fit = temp_X_fit
                X_test = temp_X_test
                X_fit_test = temp_X_fit_test
                X_fit_eval_params = temp_X_fit_eval_params

    X_fit = X_fit[:, 1:]
    X_fit_test = X_fit_test[:, 1:]
    X_fit_eval_params = X_fit_eval_params[:, 1:]

    return X_fit, X_test, X_fit_test, X_fit_eval_params
