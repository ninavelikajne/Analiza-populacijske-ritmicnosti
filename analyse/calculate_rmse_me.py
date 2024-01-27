import numpy as np
import pandas as pd

from fit import helpers as hlp, dproc
from generate_data import stationary_data as st

X_test = np.linspace(0, 100, 1000)

Y_1 = st.og_symmetric_oscillatory(X_test, period1=24, period2=24, A1=3, A2=3)
Y_2 = st.og_symmetric_oscillatory(X_test, period1=24, period2=12, A1=3, A2=3)
Y_3 = st.og_symmetric_oscillatory(X_test, period1=6, period2=8, A1=3, A2=2)

res_1 = dproc.evaluate_rhythm_params(X_test, Y_1)
res_2 = dproc.evaluate_rhythm_params(X_test, Y_2)
res_3 = dproc.evaluate_rhythm_params(X_test, Y_3)

og_df = pd.read_csv('../results/sep_classification_me.csv')
df = og_df[og_df.data_name != 'symmetric_non_oscillatory']
df = df[df.data_name != 'asymmetric_non_oscillatory']
df_results = pd.DataFrame()

for index, row in df.iterrows():
    Y_test = row['Y_test'].split('[')[1].split(']')[0]
    Y_test = np.fromstring(Y_test, sep=' ')

    y_1_actual = Y_1
    y_2_actual = Y_2
    y_3_actual = Y_3
    if row['method'] == 'cosinor_1':
        X_test = np.linspace(0, 23, 100)
        y_1_actual = st.og_symmetric_oscillatory(X_test, period1=24, period2=24, A1=3, A2=3)
        y_2_actual = st.og_symmetric_oscillatory(X_test, period1=24, period2=12, A1=3, A2=3)
        y_3_actual = st.og_symmetric_oscillatory(X_test, period1=6, period2=8, A1=3, A2=2)

    if row['amplitude'] < 0.1:  # no rhythm
        amp_rmse = np.nan
        phase_rmse = np.nan
        mesor_rmse = np.nan
        y_fit_rmse = np.nan
    else:
        if row['data_name'] == 'symmetric_oscillatory':
            amp_rmse = hlp.calculate_rmse([res_1['amplitude']], [row['amplitude']])
            phase_rmse = hlp.calculate_rmse([res_1['acrophase']], [row['acrophase']])
            mesor_rmse = hlp.calculate_rmse([res_1['mesor']], [row['mesor']])
            y_fit_rmse = hlp.calculate_rmse(y_1_actual, Y_test, norm=True)
        elif row['data_name'] == 'asymmetric_oscillatory_1':
            amp_rmse = hlp.calculate_rmse([res_2['amplitude']], [row['amplitude']])
            phase_rmse = hlp.calculate_rmse([res_2['acrophase']], [row['acrophase']])
            mesor_rmse = hlp.calculate_rmse([res_2['mesor']], [row['mesor']])
            y_fit_rmse = hlp.calculate_rmse(y_2_actual, Y_test, norm=True)
        elif row['data_name'] == 'asymmetric_oscillatory_2':
            amp_rmse = hlp.calculate_rmse([res_3['amplitude']], [row['amplitude']])
            phase_rmse = hlp.calculate_rmse([res_3['acrophase']], [row['acrophase']])
            mesor_rmse = hlp.calculate_rmse([res_3['mesor']], [row['mesor']])
            y_fit_rmse = hlp.calculate_rmse(y_3_actual, Y_test, norm=True)

    temp = row[['method', 'data_name', 'data_params', 'time_span', 'num_of_period', 'step', 'replicates', 'noise',
                'population_id', 'Y_test', 'repetition', 'changing_param']]
    temp['amplitude_rmse'] = amp_rmse
    temp['acrophase_rmse'] = phase_rmse
    temp['mesor_rmse'] = mesor_rmse
    temp['y_fit_nrmse'] = y_fit_rmse
    temp = temp.to_dict()
    df_results = pd.concat([df_results, (pd.DataFrame.from_dict(temp, orient='index')).T], ignore_index=True)

df_results = df_results.sort_values(by='y_fit_nrmse')
df_results.to_csv('../results/rmse_me.csv', index=False)
