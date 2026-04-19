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

# %% tags=["setup"]
############ LOAD in custom packages ################
import sys
import os
import pandas as pd
import numpy as np
import importlib
from fastdtw import dtw
import seaborn as sns
import matplotlib.pyplot as plt
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



################ DEFINE column variables from data ###################
from scripts.variables import id_columns, df_names, demographic_columns, daily_cols_v1, daily_cols_v2, daily_v2_common, daily_v2_weather
from scripts.variables import phq2_cols, phq9_cols, weekly_cols, passive_cols, baseline_cols, created_cols
demo_df = pd.read_csv(os.path.join(brighten_dir, f'demographics_clean.csv'))
demographic_vars = [col for col in demo_df.columns if col!='num_id' and 'Unnamed' not in col]
print(demographic_vars)
id_columns.append('idx')
time_cols = []

# %%
for name in ['v1_day']:
	df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_imputed.csv'))


# %%
daily_cols_v2

# %%
#### Take the average of each subject's symptom correlation and use it to make clusters of variables
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

symptom_matrices_df_dict = {}
flattened_matrices_df_dict = {}
target_cols = phq2_cols+phq9_cols
non_pca_cols = {}

# Create dicts of all subjects' correlation matrices
for name in ['v1_day','v1_week','v2_day','v2_week']:
	df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_imputed.csv'))
	
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
	df = df.loc[:, ~df.columns.str.contains('_indicator')]
	df = df.loc[:, ~df.columns.str.contains('_nan')]
	# take out time cols, nonnumeric cols, baseline cols
	numeric_cols = df.select_dtypes(include=('int64','float64')).columns.to_list()
	nonnumeric_cols = [col for col in df.columns if col not in numeric_cols]
	time_cols = [col for col in df.columns if 'season' in col or 'dt' in col]
	fully_empty_cols = [col for col in df.columns if 'hours' in col]
	rando_cols = [col for col in df.columns if 'user_phone_type' in col or 'race' in col or 'cohort' in col or 'marital_status' in col or 'gender' in col or 'Akili' in col]
	non_pca_cols[name] = weekly_cols+demographic_vars+time_cols+id_columns+nonnumeric_cols+baseline_cols+rando_cols+created_cols+fully_empty_cols
	
	column_order_all = daily_cols_v1 + daily_cols_v2
	column_order = [col for col in column_order_all if col in df.columns]
	print(f'\n\nDATAFRAME: {name}')
	display(df[column_order].head())
	print(column_order)
	symptom_matrix_dict, flattened_matrix_dict = fs.make_symptom_matrices(df, ignore_cols=non_pca_cols[name], num_to_plot=1, column_order=column_order)
	symptom_matrices_df_dict[name] = symptom_matrix_dict
	flattened_matrices_df_dict[name] = flattened_matrix_dict



# %%
column_order

# %%
#### Take the average of each subject's symptom correlation and use it to make clusters of variables
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

symptom_matrices_df_dict = {}
flattened_matrices_df_dict = {}
sensor_matrices_df_dict = {}
target_matrices_df_dict = {}
flattened_sensor_df_dict = {}
flattened_target_df_dict = {}

# Create dicts of all subjects' correlation matrices
for name in df_names:
	df=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_imputed.csv'))
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

	print(f'\n\nDATAFRAME: {name}')
	column_order_all =  daily_cols_v1 + daily_cols_v2
	column_order = [col for col in column_order_all if col in df.columns]

	# Make symptom/flattened dicts for all cols
	symptom_matrix_dict, flattened_matrix_dict = fs.make_symptom_matrices(df, ignore_cols=non_pca_cols[name], num_to_plot=0, column_order=column_order)

	# Assign dicts to nested dict
	symptom_matrices_df_dict[name] = symptom_matrix_dict
	flattened_matrices_df_dict[name] = flattened_matrix_dict


	# Make separate symptom/flattened dicts for target cols vs. sensor cols
	if 'week' in name:
		target_cols = phq9_cols
		sensor_matrices_dict, flattened_sensor_dict = fs.make_symptom_matrices(df, ignore_cols=non_pca_cols[name], num_to_plot=1)
		non_target_df = df.copy().drop(columns=[col for col in df.columns if col not in target_cols and 'num_id' not in col])
		target_matrices_dict, flattened_target_dict = fs.make_symptom_matrices(non_target_df, ignore_cols=non_pca_cols[name], num_to_plot=1)

	if 'day' in name:
		target_cols = phq2_cols
		sensor_matrices_dict, flattened_sensor_dict = fs.make_symptom_matrices(df, ignore_cols=non_pca_cols[name]+weekly_cols, num_to_plot=1)
		non_target_df = df.copy().drop(columns=[col for col in df.columns if col not in target_cols and 'num_id' not in col])
		target_matrices_dict, flattened_target_dict = fs.make_symptom_matrices(non_target_df, ignore_cols=non_pca_cols[name], num_to_plot=1)

	# Assign sensor/target dicts to nested dict
	sensor_matrices_df_dict[name] = sensor_matrices_dict
	target_matrices_df_dict[name] = target_matrices_dict
	flattened_sensor_df_dict[name] = flattened_sensor_dict
	flattened_target_df_dict[name] = flattened_target_dict
	



# %%
############### Plot individual networks of symptom matrices #################
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

for name, df_dict in symptom_matrices_df_dict.items():
	count=0
	for sub, corr_matrix in df_dict.items():
		if count<1:
			fs.plot_network(corr_matrix, title=f'{name}: sub {sub}')
			count=+1

# %%
# Double check where 'labels' are for columns
symptom_matrices_df_dict['v1_week'][list(symptom_matrices_df_dict['v1_week'].keys())[0]].index

# %% [markdown]
# ### Ok so temp mean and dew point IQR keep having relatively high correlations with the target variables, so we'll keep these, and drop the rest of the weather variables. Humidity also does but since it's highyl correlated with temp, we'll keep it out. 

# %% [markdown]
# # SENSORS PCA

# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

## Plot options
for name, df in sensor_matrices_df_dict.items():
	avg_matrix = fs.average_matrix(df)
	fs.plot_hier_agg(avg_matrix, df[list(df.keys())[0]].index, is_dict=False, group_title=f'{name}') # 3, ward


# %% [markdown]
# ### Mark down best fits:
# * v1_day -> 5, single
# * v2_day -> 8, single
# * v1_week -> 6, average
# * v2_week ->9, average

# %%
################# Create cluster_dict of Variables + Cluster Labels ############
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))
importlib.reload(fs)

sensor_clust_dict = {}
# clust_dict['v1_day'] = v1_day_clust_dict
for name, dictionary in sensor_matrices_df_dict.items():
	avg_matrix = fs.average_matrix(dictionary)
	print(f'For {name}...')
	if 'v1_day' in name:
		sensor_clust_dict[name] = fs.hier_agg_clustering(avg_matrix, dictionary[list(dictionary.keys())[0]].index, n_clusters=5, linkage='average', is_dict=False)
	if 'v2_day' in name:
	   sensor_clust_dict[name] = fs.hier_agg_clustering(avg_matrix, dictionary[list(dictionary.keys())[0]].index, n_clusters=9, linkage='single', is_dict=False)
	if 'v1_week' in name:
		sensor_clust_dict[name] = fs.hier_agg_clustering(avg_matrix, dictionary[list(dictionary.keys())[0]].index, n_clusters=6, linkage='average', is_dict=False)
	if 'v2_week' in name:
	   sensor_clust_dict[name] = fs.hier_agg_clustering(avg_matrix, dictionary[list(dictionary.keys())[0]].index, n_clusters=9, linkage='single', is_dict=False)

# %%
v1_day_trainval_imputed =pd.read_csv(os.path.join(brighten_dir, 'v1_day_trainval_imputed.csv'))
v1_week_trainval_imputed =pd.read_csv(os.path.join(brighten_dir, 'v1_week_trainval_imputed.csv'))
v2_day_trainval_imputed =pd.read_csv(os.path.join(brighten_dir, 'v2_day_trainval_imputed.csv'))
v2_week_trainval_imputed =pd.read_csv(os.path.join(brighten_dir, 'v2_week_trainval_imputed.csv'))


# %%
################# Create df of Variables + 1st PC of clusters Labels ############
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))
importlib.reload(fs)
dfs_pca = {}
dfs_pca_dict = {}
test_pca = {}
test_pca_dict = {}
for name in df_names:
	print(f'\nFor {name}')

	df=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_imputed.csv'))
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
	dfs_pca[name], dfs_pca_dict[name] = fs.pca_on_clusters(df, sensor_clust_dict[name], n_clusters=13) #n_clusters = number shown, so choose max
	
    # Make test df pca with the same clusters as defined by trainval, so we can use for prediction
	test_df=pd.read_csv(os.path.join(brighten_dir, f'{name}_test_imputed.csv'))
	test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]
	test_pca[name], test_pca_dict[name] = fs.pca_on_clusters(test_df, sensor_clust_dict[name], n_clusters=13) #n_clusters = number shown, so choose max


# %%

# %%
# Create PCA loadings for Trainval DFs
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

# v1_day
dfs_pca_dict['v1_day'][0]['name'] = 'pc_mobility'
dfs_pca_dict['v1_day'][1]['name'] = 'pc_calls'
dfs_pca_dict['v1_day'][2]['name'] = 'pc_unreturned_calls'
dfs_pca_dict['v1_day'][3]['name'] = 'pc_sms'
dfs_pca_dict['v1_day'][4]['name'] = 'pc_phq2'

# v2_day
dfs_pca_dict['v2_day'][0]['name'] = 'pc_commute' # transit/high speed
dfs_pca_dict['v2_day'][1]['name'] = 'pc_active_mobility'
dfs_pca_dict['v2_day'][2]['name'] = 'pc_cloud_cover_std'
dfs_pca_dict['v2_day'][3]['name'] = 'pc_weather_means' # humidity, cloud cover, precip
dfs_pca_dict['v2_day'][4]['name'] = 'pc_temp_humidity_std'
dfs_pca_dict['v2_day'][5]['name'] = 'pc_dew_point'
dfs_pca_dict['v2_day'][6]['name'] = 'pc_dew_point_std'
dfs_pca_dict['v2_day'][7]['name'] = 'pc_temp'
dfs_pca_dict['v2_day'][8]['name'] = 'pc_phq2'

# v1_week
dfs_pca_dict['v1_week'][0]['name'] = 'pc_unreturned_calls'
dfs_pca_dict['v1_week'][1]['name'] = 'pc_calls'
dfs_pca_dict['v1_week'][2]['name'] = 'pc_mobility'
dfs_pca_dict['v1_week'][3]['name'] = 'pc_phq2'
dfs_pca_dict['v1_week'][4]['name'] = 'pc_sms'
dfs_pca_dict['v1_week'][5]['name'] = 'pc_mobility_radius'

# v2_week
dfs_pca_dict['v2_week'][0]['name'] = 'pc_active_mobility'
dfs_pca_dict['v2_week'][1]['name'] = 'pc_commute'
dfs_pca_dict['v2_week'][2]['name'] = 'pc_temp_dew_point'
dfs_pca_dict['v2_week'][3]['name'] = 'pc_humidity_cloud_mean'
dfs_pca_dict['v2_week'][4]['name'] = 'pc_cloud_cover_std'
dfs_pca_dict['v2_week'][5]['name'] = 'pc_dew_point_std'
dfs_pca_dict['v2_week'][6]['name'] = 'pc_phq2'
dfs_pca_dict['v2_week'][7]['name'] = 'pc_temp_humidity_std'
dfs_pca_dict['v2_week'][8]['name'] = 'pc_precip'


for name in df_names:
	df_scaled = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_imputed.csv'))
	dfs_pca[name] = fs.merge_df_via_cluster_pca_dict(df_scaled, dfs_pca_dict[name], on_columns=['num_id', 'day'])
	dfs_pca[name].to_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'), index=False)



# %%
# Create PCA loadings for Test DFs
# But its' not getting visualized anywhere
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

# v1_day
test_pca_dict['v1_day'][0]['name'] = 'pc_mobility'
test_pca_dict['v1_day'][1]['name'] = 'pc_calls'
test_pca_dict['v1_day'][2]['name'] = 'pc_unreturned_calls'
test_pca_dict['v1_day'][3]['name'] = 'pc_sms'
test_pca_dict['v1_day'][4]['name'] = 'pc_phq2'

# v2_day
test_pca_dict['v2_day'][0]['name'] = 'pc_commute' # transit/high speed
test_pca_dict['v2_day'][1]['name'] = 'pc_active_mobility'
test_pca_dict['v2_day'][2]['name'] = 'pc_cloud_cover_std'
test_pca_dict['v2_day'][3]['name'] = 'pc_weather_means' # humidity, cloud cover, precip
test_pca_dict['v2_day'][4]['name'] = 'pc_temp_humidity_std'
test_pca_dict['v2_day'][5]['name'] = 'pc_dew_point'
test_pca_dict['v2_day'][6]['name'] = 'pc_dew_point_std'
test_pca_dict['v2_day'][7]['name'] = 'pc_temp'
test_pca_dict['v2_day'][8]['name'] = 'pc_phq2'

# v1_week
test_pca_dict['v1_week'][0]['name'] = 'pc_unreturned_calls'
test_pca_dict['v1_week'][1]['name'] = 'pc_calls'
test_pca_dict['v1_week'][2]['name'] = 'pc_mobility'
test_pca_dict['v1_week'][3]['name'] = 'pc_phq2'
test_pca_dict['v1_week'][4]['name'] = 'pc_sms'
test_pca_dict['v1_week'][5]['name'] = 'pc_mobility_radius'

# v2_week
test_pca_dict['v2_week'][0]['name'] = 'pc_active_mobility'
test_pca_dict['v2_week'][1]['name'] = 'pc_commute'
test_pca_dict['v2_week'][2]['name'] = 'pc_temp_dew_point'
test_pca_dict['v2_week'][3]['name'] = 'pc_humidity_cloud_mean'
test_pca_dict['v2_week'][4]['name'] = 'pc_cloud_cover_std'
test_pca_dict['v2_week'][5]['name'] = 'pc_dew_point_std'
test_pca_dict['v2_week'][6]['name'] = 'pc_phq2'
test_pca_dict['v2_week'][7]['name'] = 'pc_temp_humidity_std'
test_pca_dict['v2_week'][8]['name'] = 'pc_precip'


for name in df_names:
	test_df_scaled = pd.read_csv(os.path.join(brighten_dir, f'{name}_test_imputed.csv'))
	test_pca[name] = fs.merge_df_via_cluster_pca_dict(test_df_scaled, test_pca_dict[name], on_columns=['num_id', 'day'])
	test_pca[name].to_csv(os.path.join(brighten_dir, f'{name}_test_pca.csv'), index=False)




# %%
############ Plot subjects' individual networks of PCs ############
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))


fixed_positions_v1_day = {
    'pc_mobility': (1, 1),
    'pc_calls': (1, 2),
    'pc_unreturned_calls': (1, 3),
    'pc_sms': (2, 1),
    'pc_phq2': (2, 2)
}

fixed_positions_v2_day = {
    'pc_commute': (1, 1),
    'pc_active_mobility': (1, 2),
    'pc_cloud_cover_std': (1, 3),
    'pc_weather_means': (2, 1),
    'pc_temp_humidity_std': (2, 2),
    'pc_dew_point': (2, 3),
    'pc_dew_point_std': (3, 1),
    'pc_temp': (3, 2),
    'pc_phq2': (3, 3)
}

fixed_positions_v1_week = {
    'pc_unreturned_calls': (1, 1),
    'pc_calls': (1, 2),
    'pc_mobility': (1, 3),
    'pc_phq2': (2, 1),
    'pc_sms': (2, 2),
    'pc_mobility_radius': (2, 3)
}

fixed_positions_v2_week = {
    'pc_active_mobility': (1, 1),
    'pc_commute': (1, 2),
    'pc_temp_dew_point': (1, 3),
    'pc_humidity_cloud_mean': (2, 1),
    'pc_cloud_cover_std': (2, 2),
    'pc_dew_point_std': (2, 3),
    'pc_phq2': (3, 1),
    'pc_temp_humidity_std': (3, 2),
    'pc_precip': (3, 3)
}

for name, df in dfs_pca.items():
	count, count2 = 0, 0
	
	cols=[col for col in df.columns.to_list() if 'pc_' in col]
	for sub in df['num_id'].unique():
		sub_df = df[df['num_id']==sub]
		corr_matrix=sub_df[cols].corr()
		if sub_df.shape[0] > 8:
			if count<7:
				if 'v1_day' in name:
					fs.plot_network(corr_matrix, title=f'{name}: sub {sub}', threshold=0.2, fixed_positions=fixed_positions_v1_day)
					count+=1
			if count2<7:
				if 'v1_week' in name:
					fs.plot_network(corr_matrix, title=f'{name}: sub {sub}', threshold=0.2, fixed_positions=fixed_positions_v1_week)
					count2+=1


# %% [markdown]
#

# %% [markdown]
# ## Checking for collinearity
#

# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

for name in df_names:
	df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))
	print(f'\n\nFor {name}:')
	numeric_cols = [col for col in df.columns.to_list() if 'pc_' in col]
	sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0)
	plt.show()

# %% [markdown]
# We can see that our features are successfull not correlated, except phq2 and phq9, which is expected, and in most cases, phq2 won't be used to predict phq9. 

# %%
# ### Create lag variable
# for name in df_names:
#     df = pd.read_csv(os.path.join(brighten_dir, f'{name}_pca.csv'))
#     print(f'\n\nFor {name}:')
#     lag = phq9_cols + phq2_cols
#     to_lag = [col for col in lag if col in df.columns]
#     df_lag = pre.create_lag_variables(df, to_lag)
#     df_lag.to_csv(os.path.join(brighten_dir, f'{name}_pca_lag.csv'), index=False)
#     print(f"Saved {name}_pca_lag.csv to brighten_dir")


