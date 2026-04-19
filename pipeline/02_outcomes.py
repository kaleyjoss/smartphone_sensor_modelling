# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import sklearn
import numpy as np
import pandas as pd
import os
import re 
import sys 
import importlib
import datetime as dt

############ LOAD in custom packages ################

project_root = os.path.join(os.getcwd(), "..") # Get path of the project 
sys.path.append(project_root) # Add project root to sys.path for script usage

# Import and reload (optional) custom scripts
from scripts import paths
from scripts import preprocessing as pre
from scripts import visualization as vis
from scripts import variables
from scripts import feature_selection as fs

importlib.reload(paths)
importlib.reload(pre)
importlib.reload(vis)
importlib.reload(variables)
importlib.reload(fs)


# Filepaths
brighten_dir = paths.DATA
# Define label variables
df_names = ['v1_day', 'v2_day', 'v1_week', 'v2_week']




# %%
# Create binary of depression score at the end of the time
count=0
start_depressed=np.nan
end_depressed=np.nan
last_phq9=np.nan
change_bin=np.nan
for name in ['v1_day','v2_day']: 
	phq9_end = []
	Xy = pd.read_csv(os.path.join(brighten_dir, f"{name}_weekFilled.csv"))
	for sub, sub_df in Xy.groupby('num_id'):
		sub_phq9 = sub_df.dropna(subset='phq9_sum')
		if len(sub_phq9) == 0:
			continue
		sub_phq9 = sub_phq9.sort_values(by='day', ascending=True)

		first_phq9 = list(sub_phq9['phq9_sum'])[0]
		if first_phq9 > 12:
			start_depressed = 1
		else:
			start_depressed = 0


		# Phq9 at ~6 weeks
		days6weeks=sub_phq9[sub_phq9['day']>38]
		days6weeks=days6weeks[days6weeks['day']<55]
		if not len(days6weeks) > 0:
			phq9_end.append([sub, first_phq9, start_depressed, last_phq9, end_depressed, change_bin]) #np.nan for last_phq9, end_depressed
			continue
		days6weeks_cols = days6weeks.dropna(how='all', axis=1)
		if not 'phq9_sum' in days6weeks_cols:
			phq9_end.append([sub, first_phq9, start_depressed, last_phq9, end_depressed, change_bin]) #np.nan for last_phq9, end_depressed
			continue

		last_phq9 = list(days6weeks['phq9_sum'])[0]
		if last_phq9 > 12:
			end_depressed = 1
		else:
			end_depressed = 0
		
		change_bin = start_depressed - end_depressed

		# if count < 3:
		# 	count+=1
		# 	print(f'Sub: {sub}')
		# 	display(days6weeks[['day','dt','phq9_sum']])
		# 	display(f'day: {list(days6weeks['day'])[0]}, phq9: {list(days6weeks['phq9_sum'])[0]}')
			

		phq9_end.append([sub, first_phq9, start_depressed, last_phq9, end_depressed, change_bin])

	phq9_end_df = pd.DataFrame(phq9_end, columns=['num_id', 'phq9_sum_start', 'start_depressed_binary', 'phq9_sum_6wks', '6wks_depressed_binary', 'depression_change_bin'])

	phq9_end_df.to_csv(os.path.join(brighten_dir, f'{name}_phq9sum_6wks.csv'), index=False)
	display(phq9_end_df)
	print(f'Saved phq9_end_df to {name}_phq9sum_6wks.csv')


print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

# %%
v1_phq9_end_df=pd.read_csv(os.path.join(brighten_dir, f'v1_day_phq9sum_6wks.csv'))
v2_phq9_end_df=pd.read_csv(os.path.join(brighten_dir, f'v2_day_phq9sum_6wks.csv'))
phq9_end_df = pd.concat([v1_phq9_end_df, v2_phq9_end_df], axis=0)
phq9_end_df = phq9_end_df.loc[:, ~phq9_end_df.columns.str.contains('Unnamed')]
display(phq9_end_df)

phq9_end_df.to_csv(os.path.join(brighten_dir, f'phq9sum_6wks.csv'), index=False)

# %%
# Merge in 
for name in df_names:
    df = pd.read_csv(os.path.join(brighten_dir, f'{name}_weekFilled.csv'), low_memory=False)

    if 'v1' in name:
        phq9_change = pd.read_csv(os.path.join(brighten_dir, 'v1_day_phq9sum_6wks.csv')).rename(columns={'week':'ending_week'})
    else:
        phq9_change = pd.read_csv(os.path.join(brighten_dir, 'v2_day_phq9sum_6wks.csv')).rename(columns={'week':'ending_week'})

    phq9_change=phq9_change.drop(columns=[col for col in phq9_change.columns if 'Unnamed' in col])
    merge_df = df.merge(phq9_change, on=['num_id'],how='outer')
    merge_df.to_csv(os.path.join(brighten_dir, f'{name}_outcomes.csv'), index=False)

print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))


# %%
for name in df_names:
    merge_df.to_csv(os.path.join(brighten_dir, f'{name}_outcomes.csv'), index=False)
	
