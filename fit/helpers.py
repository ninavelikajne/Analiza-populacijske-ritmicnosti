import random

import numpy as np
import pandas as pd
import scipy.stats as stats

param_dict = {'noise': 'šum n', 'num_of_period': 'časovno obdobje zbiranja podatkov p',
              'step': 'časovni korak zbiranja podatkov s', 'replicates': ' število subjektov v populaciji r'}


def acro_circ(df, combo):
    valid = df[df.parameter == combo]
    mean = stats.circmean(list(valid['acrophase'].to_numpy()), low=-2 * np.pi, high=0)
    std = stats.circstd(list(valid['acrophase'].to_numpy()), low=-2 * np.pi, high=0)

    return mean, std


# project acrophase to the interval [-pi, pi]
def project_acr(acr):
    acr %= (2 * np.pi)
    if acr > np.pi:
        acr -= 2 * np.pi
    elif acr < -np.pi:
        acr += 2 * np.pi
    return acr


def clean_data(df, x, y):
    df = df.dropna(subset=[x, y])
    X = df[x].unique()

    for hour in X:
        df_hour = df.loc[df[x] == hour].copy()
        # cleaning outliers
        df_hour = df_hour.loc[df_hour[y] >= df_hour[y].quantile(0.15)].copy()
        df_hour = df_hour.loc[df_hour[y] <= df_hour[y].quantile(0.85)].copy()
        df.loc[df[x] == hour, [y]] = df_hour[y]

    df = df.dropna(subset=[x, y])
    return df


def clean_data_id(df, x, y, id):
    models = ['1', '2', '3', '4', '5']
    df = df.dropna(subset=[x, y])
    X = df[x].unique()
    ids = df[id].unique()
    final = -1

    for ix in ids:
        df_new = df[df[id] == ix]
        df_new['model'] = random.choice(models)
        if df_new.shape[0] > 10:
            for hour in X:
                df_hour = df_new.loc[df_new[x] == hour].copy()
                # cleaning outliers
                df_hour = df_hour.loc[df_hour[y] >= df_hour[y].quantile(0.15)].copy()
                df_hour = df_hour.loc[df_hour[y] <= df_hour[y].quantile(0.85)].copy()
                # df.loc[df[x] == hour and df[id]==ix, [y]] = df_hour[y]

                if type(final) == int:
                    final = df_hour.copy()
                else:
                    final = pd.concat([final, df_hour], axis=0, ignore_index=True)

    # df = df.dropna(subset=[x, y])
    return final


def amp_acr(beta_s, beta_r, corrected=True):
    amp = (beta_s ** 2 + beta_r ** 2) ** (1 / 2)

    # print("rrr (beta)", beta_r, "sss (gamma):", beta_s)

    if corrected:

        if type(beta_s) != np.ndarray:
            beta_s = np.array([beta_s])
            beta_r = np.array([beta_r])

        acr = np.zeros(len(beta_s))
        # acr corrected according to cosinor2
        for i in range(len(beta_s)):
            rrr = beta_r[i]
            sss = beta_s[i]

            if (rrr > 0) and (sss > 0):
                acr[i] = 0 + (-1 * np.arctan(np.abs(sss / rrr)))
                # acr[i] = np.arctan(sss / rrr)
            elif (rrr > 0) and (sss < 0):
                acr[i] = 2 * (-1) * np.pi + (1 * np.arctan(np.abs(sss / rrr)))
                # acr[i] = (1*np.arctan(sss / rrr))
            elif (rrr < 0) and (sss > 0):
                acr[i] = np.pi * (-1) + (1 * np.arctan(np.abs(sss / rrr)))
                # acr[i] = np.pi*(-1) + (1*np.arctan(sss / rrr))
            else:
                acr[i] = np.pi * (-1) + (-1 * np.arctan(np.abs(sss / rrr)))
                # acr[i] = np.pi + np.arctan(sss / rrr)

            # acr[i] %= 2*np.pi
            # if acr[i] < 0:
            #    acr[i] = acr[i] + 2*np.pi
            # print(acr)
        if type(amp) != np.ndarray:
            acr = acr[0]

            # acr = np.arctan2(beta_s, beta_r)
        # acr = np.arctan2(beta_r, beta_s)
        # acr = np.abs(acr)


    else:
        acr = np.arctan(beta_s / beta_r)

    return amp, acr


def phase_to_radians(phase, period=24):
    phase_rads = (-(phase / period) * 2 * np.pi) % (2 * np.pi)
    if phase_rads > 0:
        phase_rads -= 2 * np.pi
    return phase_rads


def fractional_part(x):
    return x % 1


def savitzky_golay(data, kernel=11, order=4):
    """
        applies a Savitzky-Golay filter
        input parameters:
        - data => data as a 1D numpy array
        - kernel => a positiv integer > 2*order giving the kernel size
        - order => order of the polynomal
        returns smoothed data as a numpy array

        invoke like:
        smoothed = savitzky_golay(<rough>, [kernel = value], [order = value]
    """
    try:
        kernel = abs(int(kernel))
        order = abs(int(order))
    except ValueError as msg:
        raise ValueError("kernel and order have to be of type int (floats will be converted).")
    if kernel % 2 != 1 or kernel < 1:
        raise TypeError("kernel size must be a positive odd number, was: %d" % kernel)
    if kernel < order + 2:
        raise TypeError("kernel is to small for the polynomals\nshould be > order + 2")

    # a second order polynomal has 3 coefficients
    order_range = range(order + 1)
    half_window = (kernel - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    # since we don't want the derivative, else choose [1] or [2], respectively
    m = np.linalg.pinv(b).A[0]
    window_size = len(m)
    half_window = (window_size - 1) // 2

    # precompute the offset values for better performance
    offsets = range(-half_window, half_window + 1)
    offset_data = zip(offsets, m)

    smooth_data = list()

    # temporary data, extended with a mirror image to the left and right
    firstval = data[0]
    lastval = data[len(data) - 1]

    # left extension: f(x0-x) = f(x0)-(f(x)-f(x0)) = 2f(x0)-f(x)
    # right extension: f(xl+x) = f(xl)+(f(xl)-f(xl-x)) = 2f(xl)-f(xl-x)
    leftpad = np.zeros(half_window) + 2 * firstval
    rightpad = np.zeros(half_window) + 2 * lastval
    leftchunk = data[1:1 + half_window]
    leftpad = leftpad - leftchunk[::-1]
    rightchunk = data[len(data) - half_window - 1:len(data) - 1]
    rightpad = rightpad - rightchunk[::-1]
    data = np.concatenate((leftpad, data))
    data = np.concatenate((data, rightpad))
    for i in range(half_window, len(data) - half_window):
        value = 0.0
        for offset, weight in offset_data:
            value += weight * data[i + offset]
        smooth_data.append(value)
    return np.array(smooth_data)


def f_test(first_row, second_row):
    n_components1 = first_row['method_params']['n_components']
    n_components2 = second_row['method_params']['n_components']

    n_points = len(first_row['Y_est'])
    RSS1 = first_row.RSS
    RSS2 = second_row.RSS
    DF1 = n_points - (n_components1 * 2 + 1)
    DF2 = n_points - (n_components2 * 2 + 1)

    if DF2 < DF1:
        F = ((RSS1 - RSS2) / (DF1 - DF2)) / (RSS2 / DF2)
        f = 1 - stats.f.cdf(F, DF1 - DF2, DF2)
    else:
        F = ((RSS2 - RSS1) / (DF2 - DF1)) / (RSS1 / DF1)
        f = 1 - stats.f.cdf(F, DF2 - DF1, DF1)

    if f < 0.05:
        return second_row

    return first_row


def get_best_n_components(df_results):
    df_results = df_results[df_results['method'] == 'cosinor'].copy()

    i = 0
    for index, new_row in df_results.iterrows():
        if i == 0:
            best_row = new_row
            i = 1
        else:
            best_row = f_test(best_row, new_row)

    return best_row


# convert phase angles in radians to time units
def acrophase_to_hours(acrophase, period=24):
    acrophase = project_acr(acrophase)
    hours = -period * acrophase / (2 * np.pi)
    if hours < 0:
        hours += period
    return hours


def cosinor1_classify_rhythm(rhythm, p_amp, amp_threshold=0.05):
    str = ''
    tp, fp, tn, fn = 0, 0, 0, 0
    if rhythm == 1 and p_amp < amp_threshold:
        tp = 1
        str = 'tp'
    elif rhythm == 1 and p_amp > amp_threshold:
        fn = 1
        str = 'fn'
    elif rhythm == 0 and p_amp < amp_threshold:
        fp = 1
        str = 'fp'
    elif rhythm == 0 and p_amp > amp_threshold:
        tn = 1
        str = 'tn'
    return tp, fp, tn, fn, str


def classify_rhythm(rhythm, amplitude, amp_threshold=0.1):
    str = ''
    tp, fp, tn, fn = 0, 0, 0, 0
    if rhythm == 1 and amplitude > amp_threshold:
        tp = 1
        str = 'tp'
    elif rhythm == 1 and amplitude < amp_threshold:
        fn = 1
        str = 'fn'
    elif rhythm == 0 and amplitude > amp_threshold:
        fp = 1
        str = 'fp'
    elif rhythm == 0 and amplitude < amp_threshold:
        tn = 1
        str = 'tn'
    return tp, fp, tn, fn, str


def calculate_rmse(actual, predicted, norm=False):
    if len(actual) != len(predicted):
        raise ValueError("Input arrays must have the same length.")
    actual = np.array(actual)
    predicted = np.array(predicted)
    squared_errors = (actual - predicted) ** 2
    mean_squared_errorr = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_errorr)

    if norm:
        mini = actual.min()
        maxi = actual.max()
        rmse = rmse / (maxi - mini)

    return rmse


def precision(tp, fp):
    try:
        return tp / (tp + fp)
    except:
        return np.nan


def recall(tp, fn):
    try:
        return tp / (tp + fn)
    except:
        return np.nan


def f1_score(tp, fp, fn):
    precision_value = precision(tp, fp)
    recall_value = recall(tp, fn)
    try:
        if precision_value + recall_value == 0:
            return 0
        return 2 * (precision_value * recall_value) / (precision_value + recall_value)
    except:
        return np.nan


def get_scores(b, param, agg_results, method, data_name, repetition=0):
    a = b[b.changing_param == param]

    for param_val in a[param].unique():
        c = a[a[param] == param_val]
        cnt = c.groupby(['classification']).count()
        tp, tn, fp, fn = 0, 0, 0, 0
        for ind in cnt.index:
            if (ind == 'tp'):
                tp = tp + cnt.loc[ind]['method']
            elif (ind == 'tn'):
                tn = tn + cnt.loc[ind]['method']
            elif (ind == 'fp'):
                fp = fp + cnt.loc[ind]['method']
            elif (ind == 'fn'):
                fn = fn + cnt.loc[ind]['method']

        p = precision(tp, fp)
        r = recall(tp, fn)
        f1 = f1_score(tp, fp, fn)

        below = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (below == 0):
            mcc = (tp * tn - fp * fn) / 0.001
        else:
            mcc = (tp * tn - fp * fn) / below

        temp = {'method': method, 'data_name': data_name, 'repetition': repetition, 'param': param,
                'param_val': param_val, 'tp': tp, 'tn': tn,
                'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': f1, 'mcc': mcc}
        agg_results = pd.concat([agg_results, (pd.DataFrame.from_dict(temp, orient='index')).T],
                                ignore_index=True)
    return agg_results


def get_column_names(n_components):
    base1 = 'ss'
    base2 = 'rr'
    columns = []
    for i in range(n_components):
        temp1 = base1 + str(i + 1)
        temp2 = base2 + str(i + 1)
        columns.append(temp1)
        columns.append(temp2)
    return columns


def formulate_formula_interaction(interactions, columns):
    formula = '~'
    interaction = ''
    for i, inter in enumerate(interactions):
        if i == 0:
            interaction = interaction + inter
        else:
            interaction = interaction + "*" + inter

    formula = formula + interaction
    for column in columns:
        formula = formula + "+" + interaction + "*" + column
    return formula


def formulate_formula(variables, columns):
    formula = '~'
    for ix, var in enumerate(variables):
        if ix == 0:
            formula = formula + var
        else:
            formula = formula + "+" + var
        for column in columns:
            formula = formula + "+" + var + "*" + column
    return formula


def sample_by_id(df, id, size):
    replicates = list(df[id].unique())
    ids = random.sample(replicates, size)
    return df[df[id].isin(ids)]


def tup2str(tuple):
    s = ''
    for i, x in enumerate(tuple):
        if i == 0:
            s = s + x
        else:
            s = s + ", " + x
    return s
