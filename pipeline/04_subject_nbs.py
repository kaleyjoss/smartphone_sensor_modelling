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
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import os
import re 
import sys 
import importlib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
	OneHotEncoder,
	OrdinalEncoder,
	RobustScaler,
)
from sklearn.preprocessing import (
	PowerTransformer, 
	QuantileTransformer
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler, QuantileTransformer, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_validate
sklearn.set_config(enable_metadata_routing=False)
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from xgboost import XGBClassifier, XGBRegressor
import time 
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import pearsonr
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

import shap
shap.initjs()


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
from scripts.variables import id_columns
from scripts.variables import all_cols, all_daily_cols, weekly_cols, baseline_cols, drop_weekly_cols
from scripts.variables import daily_cols_v1, daily_v2_sensor_hr, daily_v2_weather, daily_cols_v2 
from scripts.variables import gad_cols, phq9_base, alc_cols, phq9_cols, phq2_cols, sleep_cols, gic_cols, sds_cols


# Define label variables
df_names = ['v1_day', 'v2_day', 'v1_week', 'v2_week']
aggregate_dfs = ['alldays_df','week_df']
# Update endings list if order changes


########################################## MODELS #######################################


# Models
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
# Dummy baseline model
class GroupMeanRegressor(BaseEstimator, RegressorMixin):
	def fit(self, X, y, groups):
		self.group_means_ = y.groupby(groups.squeeze()).mean().squeeze()
		self.global_mean_ = y.mean().squeeze()
		return self
	
	def predict(self, X, groups):
		return groups.squeeze().map(self.group_means_).fillna(self.global_mean_).values

# Pearson scorer
def pearsonr_scorer(y_true, y_pred):
	return pearsonr(y_true, y_pred)[0]

models = {
	'Random Forest': RandomForestRegressor(random_state=42),
	'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
	'Hist Gradient Boost': HistGradientBoostingRegressor(),
	'Group Mean': GroupMeanRegressor(),
	'Ridge': Ridge(alpha=1.0)
}
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr

def pearsonr_scorer(y_true, y_pred):
	return pearsonr(y_true, y_pred)[0]

scoring_metrics = {
	'r2': 'r2',
	'neg_mae': 'neg_mean_absolute_error',
	'neg_rmse': 'neg_root_mean_squared_error',
}



# %% [markdown]
# # NBS using each subject's symptom/sensor structure
# T test https://www.youtube.com/watch?v=VekJxtk4BYM 
#
# Needs normal distribution of the metric variables-- test for normal distribution of each corr value
#
# also the variance between my two groups must be approximately equal -- levene's test

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
#num_id	change_sum	week	change_binary	phq9_mean	phq9_bin
#v1_phq9_sum_change.csv
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

# SENSOR AND SURVEY 
# Network based statistics for high phq9 vs. low phq9 -- OVER WHOLE STUDY
import bct 
from bct import nbs
from scipy import stats

significance_level = 0.05
columns=['num_id', 'phq9_sum_start', 'start_depressed_binary', 'phq9_sum_6wks', '6wks_depressed_binary', 'depression_change_bin']

for name in ['v1_day','v2_day']:
	print(f'############################### {name} ###############################')
	data_df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_imputed.csv'))
	depressed = data_df[data_df['start_depressed_binary']==1]['num_id']

	print(f'There are {len(depressed)} subs in the depressed group')
	depressed_present = [sub for sub in depressed if sub in symptom_matrices_df_dict[name]]
	print(f'There are {len(depressed_present)} subs present from the depressed group')
	X = {key: symptom_matrices_df_dict[name][key] for key in depressed_present}
	x_arr = np.stack([df.values for df in X.values()], axis=-1)
	print(f'x_arr shape {x_arr.shape}')
		
	nondepressed = data_df[data_df['start_depressed_binary']==0]['num_id']
	print(f'There are {len(nondepressed)} subs in the nondepressed group')
	nondepressed_present = [sub for sub in nondepressed if sub in symptom_matrices_df_dict[name]]
	print(f'There are {len(nondepressed_present)} subs present from the nondepressed group')
	y = {key: symptom_matrices_df_dict[name][key] for key in nondepressed_present}
	y_arr = np.stack([df.values for df in y.values()], axis=-1)
	print(f'y_arr shape {y_arr.shape}')

	degf = len(X) + len(y) - 2
	print(f'The degrees of freedom are: {degf}')

	t_thresh = stats.t.ppf(1-significance_level, degf)
	print(f'The critical t value is {round(t_thresh, 2)}')

	pval, adj, null = nbs.nbs_bct(x_arr, y_arr, thresh=t_thresh, verbose=False)
	print(pval)
	# Convert adj to boolean mask
	mask = adj.astype(bool)
	component_cols = y[next(iter(y))].loc[:, mask]
	print(component_cols.columns.to_list())


# %%
#num_id	change_sum	week	change_binary	phq9_mean	phq9_bin
#v1_phq9_sum_change.csv
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

# SENSOR AND SURVEY 
# Network based statistics for high phq9 vs. low phq9 -- OVER WHOLE STUDY
import bct 
from bct import nbs
from scipy import stats

significance_level = 0.05
columns=['num_id', 'phq9_sum_start', 'start_depressed_binary', 'phq9_sum_6wks', '6wks_depressed_binary', 'depression_change_bin']

for name in ['v1_day','v2_day']:
	print(f'############################### {name} ###############################')
	data_df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_transformed.csv'))
	depressed = data_df[data_df['6wks_depressed_binary']==1]['num_id']

	print(f'There are {len(depressed)} subs in the depressed group')
	depressed_present = [sub for sub in depressed if sub in symptom_matrices_df_dict[name]]
	print(f'There are {len(depressed_present)} subs present from the depressed group')
	X = {key: symptom_matrices_df_dict[name][key] for key in depressed_present}
	x_arr = np.stack([df.values for df in X.values()], axis=-1)
	print(f'x_arr shape {x_arr.shape}')
		
	nondepressed = data_df[data_df['6wks_depressed_binary']==0]['num_id']
	print(f'There are {len(nondepressed)} subs in the nondepressed group')
	nondepressed_present = [sub for sub in nondepressed if sub in symptom_matrices_df_dict[name]]
	print(f'There are {len(nondepressed_present)} subs present from the nondepressed group')
	y = {key: symptom_matrices_df_dict[name][key] for key in nondepressed_present}
	y_arr = np.stack([df.values for df in y.values()], axis=-1)
	print(f'y_arr shape {y_arr.shape}')

	degf = len(X) + len(y) - 2
	print(f'The degrees of freedom are: {degf}')

	t_thresh = stats.t.ppf(1-significance_level, degf)
	print(f'The critical t value is {round(t_thresh, 2)}')

	pval, adj, null = nbs.nbs_bct(x_arr, y_arr, thresh=t_thresh, verbose=False)
	print(pval)
	# Convert adj to boolean mask
	mask = adj.astype(bool)
	component_cols = y[next(iter(y))].loc[:, mask]
	print(component_cols.columns.to_list())


# %%
#num_id	change_sum	week	change_binary	phq9_mean	phq9_bin
#v1_phq9_sum_change.csv
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

# SENSOR AND SURVEY 
# Network based statistics for high phq9 vs. low phq9 -- OVER WHOLE STUDY
import bct 
from bct import nbs
from scipy import stats

significance_level = 0.05
columns=['num_id', 'phq9_sum_start', 'start_depressed_binary', 'phq9_sum_6wks', '6wks_depressed_binary', 'depression_change_bin']

for name in ['v1_day','v2_day']:
	print(f'############################### {name} ###############################')
	data_df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_transformed.csv'))
	depressed = data_df[data_df['depression_change_bin']==1]['num_id']

	print(f'There are {len(depressed)} subs in the depressed group')
	depressed_present = [sub for sub in depressed if sub in symptom_matrices_df_dict[name]]
	print(f'There are {len(depressed_present)} subs present from the depressed group')
	X = {key: symptom_matrices_df_dict[name][key] for key in depressed_present}
	x_arr = np.stack([df.values for df in X.values()], axis=-1)
	print(f'x_arr shape {x_arr.shape}')
		
	nondepressed = data_df[data_df['depression_change_bin']==-1]['num_id']
	print(f'There are {len(nondepressed)} subs in the nondepressed group')
	nondepressed_present = [sub for sub in nondepressed if sub in symptom_matrices_df_dict[name]]
	print(f'There are {len(nondepressed_present)} subs present from the nondepressed group')
	y = {key: symptom_matrices_df_dict[name][key] for key in nondepressed_present}
	y_arr = np.stack([df.values for df in y.values()], axis=-1)
	print(f'y_arr shape {y_arr.shape}')

	degf = len(X) + len(y) - 2
	print(f'The degrees of freedom are: {degf}')

	t_thresh = stats.t.ppf(1-significance_level, degf)
	print(f'The critical t value is {round(t_thresh, 2)}')

	pval, adj, null = nbs.nbs_bct(x_arr, y_arr, thresh=t_thresh, verbose=False)
	print(pval)
	# Convert adj to boolean mask
	mask = adj.astype(bool)
	component_cols = y[next(iter(y))].loc[:, mask]
	print(component_cols.columns.to_list())

