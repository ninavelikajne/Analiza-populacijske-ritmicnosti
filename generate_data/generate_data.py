import pandas as pd

from generate_data import stationary_data as dt

df = pd.DataFrame()

params = ['replicates', 'noise', 'num_of_period', 'step']
param_values = {
    'noise': [0.3, 1.5, 4],
    'step': [1, 2, 4, 8],
    'replicates': [5, 10, 50],
    'num_of_period': [1, 2, 4, 10]
}

fixed_noise = 0.3
fixed_step = 1
fixed_replicates = 5
fixed_num_of_periods = 4
repetitions = 10
population_id = 0

for ix in range(repetitions):
    for param in params:
        values = param_values[param]
        for value in values:
            if param == 'noise':
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, fixed_replicates,
                                      'symmetric_non_oscillatory', param, noise=value,
                                      population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, fixed_replicates,
                                      'asymmetric_non_oscillatory', param, noise=value,
                                      population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, fixed_replicates, 'symmetric_oscillatory',
                                      param, period1=24,
                                      period2=24, A1=3, A2=3, noise=value, population_id=population_id, df=df,
                                      repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, fixed_replicates, 'asymmetric_oscillatory',
                                      param,
                                      data_name='asymmetric_oscillatory_1', period1=24, period2=12, A1=3, A2=3,
                                      noise=value, population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, fixed_replicates, 'asymmetric_oscillatory',
                                      param,
                                      data_name='asymmetric_oscillatory_2', period1=6, period2=8, A1=3, A2=3,
                                      noise=value, population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
            elif param == 'step':
                df = dt.generate_data(24, fixed_num_of_periods, value, fixed_replicates, 'symmetric_non_oscillatory',
                                      param, noise=fixed_noise,
                                      population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, value, fixed_replicates, 'asymmetric_non_oscillatory',
                                      param, noise=fixed_noise,
                                      population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, value, fixed_replicates, 'symmetric_oscillatory', param,
                                      period1=24,
                                      period2=24, A1=3, A2=3, noise=fixed_noise, population_id=population_id, df=df,
                                      repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, value, fixed_replicates, 'asymmetric_oscillatory',
                                      param,
                                      data_name='asymmetric_oscillatory_1', period1=24, period2=12, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, value, fixed_replicates, 'asymmetric_oscillatory',
                                      param,
                                      data_name='asymmetric_oscillatory_2', period1=6, period2=8, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
            elif param == 'replicates':
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, value, 'symmetric_non_oscillatory', param,
                                      noise=fixed_noise,
                                      population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, value, 'asymmetric_non_oscillatory', param,
                                      noise=fixed_noise,
                                      population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, value, 'symmetric_oscillatory', param,
                                      period1=24,
                                      period2=24, A1=3, A2=3, noise=fixed_noise, population_id=population_id, df=df,
                                      repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, value, 'asymmetric_oscillatory', param,
                                      data_name='asymmetric_oscillatory_1', period1=24, period2=12, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, value, 'asymmetric_oscillatory', param,
                                      data_name='asymmetric_oscillatory_2', period1=6, period2=8, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
            elif param == 'num_of_period':
                df = dt.generate_data(24, value, fixed_step, fixed_replicates, 'symmetric_non_oscillatory', param,
                                      noise=fixed_noise,
                                      population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, value, fixed_step, fixed_replicates, 'asymmetric_non_oscillatory', param,
                                      noise=fixed_noise,
                                      population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, value, fixed_step, fixed_replicates, 'symmetric_oscillatory', param,
                                      period1=24,
                                      period2=24, A1=3, A2=3, noise=fixed_noise, population_id=population_id, df=df,
                                      repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, value, fixed_step, fixed_replicates, 'asymmetric_oscillatory', param,
                                      data_name='asymmetric_oscillatory_1', period1=24, period2=12, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, value, fixed_step, fixed_replicates, 'asymmetric_oscillatory', param,
                                      data_name='asymmetric_oscillatory_2', period1=6, period2=8, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1

df.to_csv('./data/generated_data.csv', index=False)
