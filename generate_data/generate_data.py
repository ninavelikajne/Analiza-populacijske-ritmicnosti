import pandas as pd

from generate_data import stationary_data as dt

df=pd.DataFrame()

params=['replicates','noise','num_of_period','step']
param_values={
    'noise':[0.3,1.5,4],
    'step':[1,2,4,8],
    'replicates':[5,10,50],
    'num_of_period':[1,2,4,10]
}

fixed_noise=0.3
fixed_step=1
fixed_replicates=5
fixed_num_of_periods=4
repetitions=10
population_id=0

for ix in range(repetitions):
    for param in params:
        values=param_values[param]
        for value in values:
            if param=='noise':
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, fixed_replicates,
                                      'symmetric_non_oscillatory', param,noise=value,
                                      population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, fixed_replicates,
                                      'asymmetric_non_oscillatory', param,noise=value,
                                      population_id=population_id, df=df, repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, fixed_replicates, 'symmetric_oscillatory',param, period1=24,
                                      period2=24, A1=3, A2=3, noise=value, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, fixed_replicates, 'asymmetric_oscillatory',param,
                                      data_name='asymmetric_oscillatory_1', period1=24, period2=12, A1=3, A2=3,
                                      noise=value, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, fixed_replicates, 'asymmetric_oscillatory',param,
                                      data_name='asymmetric_oscillatory_2', period1=6, period2=8, A1=3, A2=3,
                                      noise=value, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
            elif param=='step':
                df = dt.generate_data(24, fixed_num_of_periods, value, fixed_replicates, 'symmetric_non_oscillatory',param,noise=fixed_noise,
                                      population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, value, fixed_replicates, 'asymmetric_non_oscillatory',param,noise=fixed_noise,
                                      population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, value, fixed_replicates, 'symmetric_oscillatory',param,
                                      period1=24,
                                      period2=24, A1=3, A2=3, noise=fixed_noise, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, value, fixed_replicates, 'asymmetric_oscillatory',param,
                                      data_name='asymmetric_oscillatory_1', period1=24, period2=12, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, value, fixed_replicates, 'asymmetric_oscillatory',param,
                                      data_name='asymmetric_oscillatory_2', period1=6, period2=8, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
            elif param=='replicates':
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, value, 'symmetric_non_oscillatory',param,noise=fixed_noise,
                                      population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, value, 'asymmetric_non_oscillatory',param,noise=fixed_noise,
                                      population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, value, 'symmetric_oscillatory',param,
                                      period1=24,
                                      period2=24, A1=3, A2=3, noise=fixed_noise, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, value, 'asymmetric_oscillatory',param,
                                      data_name='asymmetric_oscillatory_1', period1=24, period2=12, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, fixed_num_of_periods, fixed_step, value, 'asymmetric_oscillatory',param,
                                      data_name='asymmetric_oscillatory_2', period1=6, period2=8, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
            elif param=='num_of_period':
                df = dt.generate_data(24, value, fixed_step, fixed_replicates, 'symmetric_non_oscillatory',param,noise=fixed_noise,
                                      population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, value, fixed_step, fixed_replicates, 'asymmetric_non_oscillatory',param,noise=fixed_noise,
                                      population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, value, fixed_step, fixed_replicates, 'symmetric_oscillatory',param,
                                      period1=24,
                                      period2=24, A1=3, A2=3, noise=fixed_noise, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, value, fixed_step, fixed_replicates, 'asymmetric_oscillatory',param,
                                      data_name='asymmetric_oscillatory_1', period1=24, period2=12, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1
                df = dt.generate_data(24, value, fixed_step, fixed_replicates, 'asymmetric_oscillatory',param,
                                      data_name='asymmetric_oscillatory_2', period1=6, period2=8, A1=3, A2=3,
                                      noise=fixed_noise, population_id=population_id, df=df,repetition=ix)
                population_id = population_id + 1

df.to_csv('./data/generated_data.csv',index=False)

# noise_levels=[0.2,0.5,1]
# steps=[1,2,4,8]
# replicates=[5,10,50]
# num_of_periods=[1,2,4,10]
# time_span=24
# population_id=0
# for noise in noise_levels:
#     for step in steps:
#         for replicate in replicates:
#             for num_of_period in num_of_periods:
#                 df=dt.generate_data(24,num_of_period,step,replicate,'symmetric_non_oscillatory',population_id=population_id,df=df)
#                 population_id=population_id+1
#                 df = dt.generate_data(24, num_of_period, step, replicate, 'asymmetric_non_oscillatory',population_id=population_id, df=df)
#                 population_id = population_id + 1
#                 df = dt.generate_data(24, num_of_period, step, replicate, 'symmetric_oscillatory', period1=24,period2=24, A1=3, A2=3, noise=noise,population_id=population_id, df=df)
#                 population_id = population_id + 1
#                 df = dt.generate_data(24, num_of_period, step, replicate, 'asymmetric_oscillatory',data_name='asymmetric_oscillatory_1', period1=24,period2=12, A1=3, A2=3, noise=noise, population_id=population_id, df=df)
#                 population_id = population_id + 1
#                 df = dt.generate_data(24, num_of_period, step, replicate, 'asymmetric_oscillatory',data_name='asymmetric_oscillatory_2',period1=24,period2=24, A1=3, A2=3, noise=noise,population_id=population_id, df=df)
#                 population_id = population_id + 1
#
# df.to_csv('./data/generated_data.csv',index=False)

# fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# for ix in range(200):
#     Y=symmetric_stationary_non_oscillatory(X_del)
#     data_params={'missing':50,'sigma':0.1}
#     temp={'data':'symmetric_stationary_non_oscillatory','data_params':data_params,'id':ix,'X':np.array_str(X_del),'Y':np.array_str(Y),'rhythm':0}
#     df = pd.concat([df, (pd.DataFrame.from_dict(temp, orient='index')).T], ignore_index=True)
#
# plt.show()
#
# fig, ax = plt.subplots(1, 1, figsize=(12, 7), sharex=True, sharey=True)
# for ix in range(200):
#     Y=symmetric_stationary_oscillatory(X_del,3,3,18,26)
#     data_params = {'missing': 50, 'A1': 3,'A2':3,'tau1':18,'tau2':26,'sigma':0.1}
#     temp={'data':'symmetric_stationary_oscillatory','data_params':data_params,'id':ix,'X':np.array_str(X_del),'Y':np.array_str(Y),'rhythm':1}
#     df = pd.concat([df, (pd.DataFrame.from_dict(temp, orient='index')).T], ignore_index=True)
#     plot_cosopt.subplot_model(X_del, Y, X_del, Y, ax, plot_measurements=False, period=50)
# plt.show()
#
# fig, ax = plt.subplots(1, 1, figsize=(12, 7), sharex=True, sharey=True)
# for ix in range(200):
#     Y=asymmetric_stationary_non_oscillatory(X)
#     data_params = {'missing': 0, 'sigma': 1}
#     temp={'data':'asymmetric_stationary_non_oscillatory','data_params':data_params,'id':ix,'X':np.array_str(X),'Y':np.array_str(Y),'rhythm':0}
#     df = pd.concat([df, (pd.DataFrame.from_dict(temp, orient='index')).T], ignore_index=True)
#     plot_cosopt.subplot_model(X, Y, X, Y, ax, plot_measurements=False, period=50)
# plt.show()
#
# fig, ax = plt.subplots(1, 1, figsize=(12, 7), sharex=True, sharey=True)
# for ix in range(200):
#     Y=asymmetric_stationary_oscillatory(X,5,18)
#     data_params = {'missing': 0, 'A': 5,'tau': 18,'sigma': 1}
#     temp={'data':'asymmetric_stationary_oscillatory','data_params':data_params,'id':ix,'X':np.array_str(X),'Y':np.array_str(Y),'rhythm':1}
#     df = pd.concat([df, (pd.DataFrame.from_dict(temp, orient='index')).T], ignore_index=True)
#     plot_cosopt.subplot_model(X, Y, X, Y, ax, plot_measurements=False, period=50)
# plt.show()
#
# df.to_csv('./data/stationary_data.csv',index=False)
# print()
#
# #fig, ax = plt.subplots(1, 1, figsize=(12, 7), sharex=True, sharey=True)
# #plot_cosopt.subplot_model(X_del, Y, X_del, Y, ax, plot_measurements=False,period=50)
# #plt.show()