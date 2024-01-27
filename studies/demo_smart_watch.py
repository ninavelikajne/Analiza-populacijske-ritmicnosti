import pandas as pd
from fit import gee
from fit import mixed_effects as me
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

df=pd.read_csv('../data/article.csv')
df.loc[df['gender'] == 'Man', ['gender']] = 'moški'
df.loc[df['gender'] == 'Woman', ['gender']] = 'ženska'
df.loc[df['bmi_baseline_cat'] == 'Normal', ['bmi_baseline_cat']] = 'normalen'
df.loc[df['bmi_baseline_cat'] == 'Overweight', ['bmi_baseline_cat']] = 'debel'
df.loc[df['bmi_baseline_cat'] == 'Obese', ['bmi_baseline_cat']] = 'predebel'

df1=gee.gee_cosinor(df, 'hrv', 'Hour_of_Day', 'participant_id', interactions=['bmi_baseline_cat', 'gender'],n_components=2,save_to='../results/demo/gee_example.png')
df2=gee.calculate_confidence_intervals_parameters_cosinor(df, 'hrv', 'Hour_of_Day', 'participant_id', interactions=['bmi_baseline_cat', 'gender'],n_components=2,save_to='../results/demo/gee_example_ci.png')
df1.to_csv('../results/demo/gee.csv',index=False)
df2.to_csv('../results/demo/gee_ci.csv',index=False)


df3=me.me_cosinor(df, 'hrv', 'Hour_of_Day', 'participant_id', variables=['gender'],random_var='T0toT14',n_components=2,save_to='../results/demo/me_example.png')
df4=me.calculate_confidence_intervals_parameters(df, 'hrv', 'Hour_of_Day', 'participant_id', variables=['gender'],random_var='T0toT14',n_components=2,save_to='../results/demo/me_example_ci.png')
df3.to_csv('../results/demo/me.csv',index=False)
df4.to_csv('../results/demo/me_ci.csv',index=False)
