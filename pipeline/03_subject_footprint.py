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

# %% [markdown]
# ## Clustering Demographics 

# %%
######################## LOAD IN FILES #############################
import os
import pandas as pd
import sys
import importlib
import numpy as np
import pandas as pd
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import math
import plotly.express as px
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
from scripts import clustering as cl

importlib.reload(paths)
importlib.reload(pre)
importlib.reload(vis)
importlib.reload(variables)
importlib.reload(fs)
importlib.reload(cl)

# Filepaths
brighten_dir = paths.DATA
sub_dir = paths.SUB_DFS
results_dir = paths.RESULTS

################ DEFINE column variables from data ###################
from scripts.variables import id_columns, daily_cols_v1, daily_cols_v2 
from scripts.variables import phq2_cols, phq9_cols, weekly_cols, passive_cols, daily_v2_weather
from scripts.variables import df_names, df_mis, df_names_with_mis
demo_df = pd.read_csv(os.path.join(brighten_dir, f'demographics.csv'))
demographic_vars = [col for col in demo_df.columns if col!='num_id']
id_columns.append('idx')

# %% [markdown]
# ### Calculate the RMSSD for each participant on each variable - stability of each variable over time
# The root mean square of successive differences between (RMSSD) is obtained by first calculating each successive time difference (or value difference) between observations. Then, each of the values is squared and the result is averaged before the square root of the total is obtained. 
#
# The RMSSD is used as a 'stability' metric for that item for that participant.

# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

for name in ['v1_day','v2_day']:
	df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_imputed.csv'))
	print(set(df['num_id'].to_list()))

# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

correct_test_df = pd.read_csv(os.path.join(brighten_dir, 'rmssd_test.csv'))
subs_list = [1.0, 13.0, 14.0, 17.0, 29.0, 44.0, 1030.0, 519.0, 524.0, 525.0, 1040.0]
for name in ['v1_day','v2_day']:
	df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_transformed.csv'))
	df = df[[col for col in id_columns if col in df.columns.to_list()]+[col for col in df.columns.to_list() if col not in id_columns]]
	df_subs = df.where(df['num_id'].isin(subs_list))
	df_subs = df_subs.dropna(subset='num_id')
	df_subs.to_csv(os.path.join(brighten_dir, f'rmssd_{name}_subs_to_test.csv'))
	df_test = pd.DataFrame()
	# ---- Compute per-subject diffs ----
	for sub in subs_list:
		sub_df = df[df['num_id']==sub]
		if len(sub_df)>1:
			X_choices = ['cloud_cover_IQR','dew_point_IQR','humidity_IQR_indicator','phq9_1','phq2_sum']
			columns = [col for col in sub_df.columns if any(item in col for item in X_choices) and col not in id_columns and 'Unnamed' not in col and 'base' not in col]
			display(sub_df.head())
			# Duplicate all columns and shift the duplicates down by 1
			shifted = sub_df.shift(1).add_suffix('_shifted')

			# Concatenate the original and shifted columns side by side
			df_combined = pd.concat([sub_df, shifted], axis=1)

			for col in columns:
				df_combined[f'{col}_diff'] = df_combined[col] - df_combined[f'{col}_shifted']
			
			for col in columns:
				df_combined[f'{col}_sq'] = df_combined[f'{col}_diff'] * df_combined[f'{col}_diff']
			
			df_combined = df_combined[[col for col in df_combined if col in id_columns] + [col for col in df_combined.columns if any(item in col for item in columns)]]
			df_test = pd.concat([df_test, df_combined])
			
			for col in columns:
				#display(df_test[['num_id',col,f'{col}_shifted',f'{col}_diff',f'{col}_sq']])
				sq_values = df_test[f'{col}_sq'].dropna()
				rmssd_val = np.sqrt(sq_values.mean())
				print(sub, col, rmssd_val)
				
		

	df_test.to_csv(os.path.join(brighten_dir, 'test_rmssd_code.csv'), index=False)

	




# %%
import os, math
import pandas as pd
import plotly.express as px
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))


correct_test_df = pd.read_csv(os.path.join(brighten_dir, 'rmssd_test.csv'))
rmssd = []
for name in df_names:
	df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_transformed.csv'))
	df_test = pd.DataFrame()
	# ---- Compute per-subject diffs ----
	for sub, sub_df in df.groupby('num_id'):
		X_choices = weekly_cols+daily_cols_v1+daily_cols_v2
		columns = [col for col in sub_df.columns if any(item in col for item in X_choices) and col not in id_columns and 'Unnamed' not in col and 'base' not in col]
		
		# Duplicate all columns and shift the duplicates down by 1
		shifted = sub_df.shift(1).add_suffix('_shifted')

		# Concatenate the original and shifted columns side by side
		df_combined = pd.concat([sub_df, shifted], axis=1)

		for col in columns:
			df_combined[f'{col}_diff'] = df_combined[col] - df_combined[f'{col}_shifted']
		
		for col in columns:
			df_combined[f'{col}_sq'] = df_combined[f'{col}_diff'] * df_combined[f'{col}_diff']
		
		df_combined = df_combined[[col for col in df_combined if col in id_columns] + [col for col in df_combined.columns if any(item in col for item in columns)]]
		df_test = pd.concat([df_test, df_combined])
		
		for col in columns:
			#display(df_test[['num_id',col,f'{col}_shifted',f'{col}_diff',f'{col}_sq']])
			sq_values = df_test[f'{col}_sq'].dropna()
			rmssd_val = np.sqrt(sq_values.mean())
			rmssd.append([sub, col, round(rmssd_val, 3)])
	




	# ---- Create DataFrame and plot ----
	rmssd_df = pd.DataFrame(rmssd, columns=['num_id', 'column', 'RMSSD'])

	for col, col_df in rmssd_df.groupby('column'):
		
		rmssd_fig = px.scatter(
			col_df,
			x='num_id',
			y='RMSSD',
			height=500,
			width=700,
			title=f'RMSSD for each subject in {col} ({name})'
		)
		rmssd_fig.show()

	rmssd_df.to_csv(os.path.join(results_dir, f'{name}_rmssd.csv'), index=False)


# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))
demo_df = pd.read_csv(os.path.join(brighten_dir, f'demographics_clean.csv'))

for feature in ['phq9_sum']:
	for name in df_names:
		for comparison in ['phq9_2_base','phq9_1_base']:
			rmssd_df = pd.read_csv(os.path.join(results_dir, f'{name}_rmssd.csv'))
			phq9_df = rmssd_df[rmssd_df['column']==feature]
			merge_df = phq9_df.merge(demo_df, on=['num_id'])
			display(merge_df.head())

			scatter = px.scatter(merge_df, x='RMSSD', y=comparison, title=f"Comparison of RMSSD of {feature} by {comparison}")
			scatter.show()




