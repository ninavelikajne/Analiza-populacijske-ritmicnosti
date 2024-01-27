import random
import numpy as np
import pandas as pd
from fit import helpers as hlp


def og_symmetric_oscillatory(X,A1, A2, period1,period2):
    Y=A1*np.sin((2*np.pi*X)/period1)+A2*np.sin((2*np.pi*X)/period2)
    return Y

def og_asymmetric_oscillatory(X,A,period):
    Y=A*hlp.fractional_part(X/period)
    return Y

def symmetric_non_oscillatory(X, sigma=0.5, mu=3):
    Y=np.random.normal(mu, sigma, len(X))
    return Y

def asymmetric_non_oscillatory(X,noise):
    border=int(noise*15)
    Y=(np.array(random.sample(range(-border*1000, border*1000), len(X))))/1000
    return Y

def symmetric_oscillatory(X,A1, A2, period1,period2,sigma=0.1): #0.5
    Y=A1*np.sin((2*np.pi*X)/period1)+A2*np.sin((2*np.pi*X)/period2)+np.random.normal(0, sigma, len(X))
    return Y

def asymmetric_oscillatory(X,A,period,sigma=1):
    Y=A*hlp.fractional_part(X/period)+np.random.normal(0, sigma, len(X))
    return Y

def generate_x(time_span,num_of_period,step):
    X=np.array([])
    for ix in range(num_of_period):
        temp = np.arange(0, time_span,step)
        X = np.concatenate((X, temp))
    return X

def generate_data(time_span, num_of_period, step, replicates, data_type, param,data_name='',period1=24, period2=24, A1=3, A2=3, noise=0.1,population_id=1,df=pd.DataFrame(),repetition=0):
    if data_name=='':
        data_name=data_type
    X = generate_x(time_span, num_of_period, step)
    rhythm=0
    data_params={}
    for ix in range(replicates):
        if data_type=='symmetric_non_oscillatory':
            rhythm=0
            data_params={'time_span':time_span, 'num_of_period':num_of_period, 'step':step, 'replicates':replicates,'noise':noise}
            Y=symmetric_non_oscillatory(X, mu=A1,sigma=noise)
        elif data_type=='asymmetric_non_oscillatory':
            rhythm=0
            data_params={'time_span':time_span, 'num_of_period':num_of_period, 'step':step, 'replicates':replicates,'noise':noise}
            Y=asymmetric_non_oscillatory(X,noise=noise)
        elif data_type=='symmetric_oscillatory':
            rhythm=1
            data_params = {'time_span': time_span, 'num_of_period': num_of_period, 'step': step, 'replicates': replicates,
                           'noise': noise,'A1':A1,'A2':A2,'period1':period1,'period2':period2}
            Y = symmetric_oscillatory(X, A1, A2, period1, period2, noise)
        elif data_type=='asymmetric_oscillatory':
            rhythm=1
            data_params = {'time_span': time_span, 'num_of_period': num_of_period, 'step': step,
                           'replicates': replicates,'noise': noise, 'A1': A1, 'period1': period1}
            Y=symmetric_oscillatory(X, A1, A2, period1, period2, noise)

        temp=pd.DataFrame({'X':X,'Y':Y})
        temp['id']=ix
        temp['data']=data_name
        temp['data_params']=str(data_params)
        temp['rhythm']=rhythm
        temp['population_id']=population_id
        temp['repetition']=repetition
        temp['changing_param']=param

        df = pd.concat([df, temp], ignore_index=True)

    return df
