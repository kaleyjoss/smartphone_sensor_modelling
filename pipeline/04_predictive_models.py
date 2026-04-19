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
import datetime as dt
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
# import seaborn as sns
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





# %%
# Just so i can view them :3
a_v1day_pca_df = pd.read_csv(os.path.join(brighten_dir, 'v1_day_trainval_pca.csv'))
a_v1week_pca_df = pd.read_csv(os.path.join(brighten_dir, 'v1_week_trainval_pca.csv'))
a_v2day_pca_df = pd.read_csv(os.path.join(brighten_dir, 'v2_day_trainval_pca.csv'))
a_v2week_pca_df = pd.read_csv(os.path.join(brighten_dir, 'v2_week_trainval_pca.csv'))


# %% [markdown]
# ## Anova for feature analysis for all subjects

# %%
## Anova for feature analysis
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import pandas as pd
from collections import Counter

# ---- EDIT PARAMS ------
n_features = 10 # Choose number of features/scores to save from the ANOVA
n_subjects_show = 3 # Choose number of subjects' ANOVA scores to see in graphs
n_features_show = 10 # Choose number of features to see in graphs
show_means = True # Choose whether to visualize mean scores for features across all subjects
# --------------------

anova_features = {}
top_features = {}
	
for y_col in ['phq2_sum','phq9_sum']:
	anova_features[y_col] = {}
	top_features[y_col]={}

	for name in ['v1_day','v2_day','v1_week','v2_week']:
		subjects_shown_count=0
		top_features[y_col][name] = {}
		print(f'################ {name} ###############')
		df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))
		for sub, sub_df in df.groupby('num_id'):
			
			sub_df_anova = sub_df.drop(columns=[col for col in sub_df.columns if 'Unnamed' in col or '_nan' in col or '_indicator' in col or col in ['day','week','month']+id_columns])
			sub_df_anova = sub_df_anova.drop(columns=[col for col in sub_df_anova.columns if any(item in col for item in baseline_cols) or 'season' in col or 'cohort' in col or 'dt' in col or 'user_phone_type' in col or 'base' in col])
			sub_df_anova = sub_df_anova.drop(columns=[col for col in sub_df_anova.columns if col in id_columns])


			df_cov = sub_df_anova.loc[:, sub_df_anova.notna().mean() > 0.7]
			df_cov = df_cov.copy()
			if y_col not in df_cov.columns:
				df_cov[y_col] = sub_df_anova[y_col].copy()
			#print(f'Not including these columns in X, not enough coverage: ', [col for col in sub_df_anova.columns if col not in df_cov.columns])
			df_full = df_cov.dropna()
			#print(f'Shape before dropna: {df_cov.shape}, shape afer dropna: {df_full.shape}')
			min_rows = 6 if 'week' in name else 15
			if len(df_full) <= min_rows:
				continue # skip small samples

			# Separate X and y
			selector = SelectKBest(score_func=f_classif, k=2)
			if 'phq2' in y_col:
				X = df_full.drop(columns=[col for col in df_full.columns if 'phq' in col])
			if 'phq9' in y_col:
				X = df_full.drop(columns=[col for col in df_full.columns if 'phq9' in col])
			X_num = X.select_dtypes(include=('int64','float64'))
			# print(f'Not including these columns in X, not numeric: ', [col for col in X.columns if col not in X_num.columns])
			
			y = df_full[y_col].copy()
			if y.nunique() < 2 or y.var() == 0:
				print(f'Warning: 0 variance in y for sub {sub}, y = {y.unique()}')
				continue
			if y.std() < 0.1:
				print(f'Warning: low variance in y for sub {sub}, std={y.std():.3f}, y = {y.unique()}')
				continue



			# --- Drop constant (zero-variance) features ---
			var_selector = VarianceThreshold(threshold=1e-10)
			X_var = pd.DataFrame(
				var_selector.fit_transform(X_num),
				columns=X_num.columns[var_selector.get_support()],
				index=X_num.index
			)

			if X_var.shape[1] < 2:
				continue

			X_new = selector.fit_transform(X_var, y)

			# Show scores
			anova_scores = pd.DataFrame({
				'feature': X_var.columns,
				'f_score': selector.scores_,
				'p_value': selector.pvalues_
			}).sort_values('f_score', ascending=False)

			#print(anova_scores)
			
			# --- Save Features --- 
			top_n = 10  # number of top features to save
			top_features_sub = anova_scores.head(top_n)
			top_features[y_col][name][sub] = top_features_sub['feature'].to_list()

			# --- Visualize Scores for Subjects ---
			if subjects_shown_count<n_subjects_show:
				plt.figure(figsize=(10, 6))
				plt.barh(top_features_sub['feature'], top_features_sub['f_score'], color='skyblue')
				plt.xlabel('F-score (ANOVA)')
				plt.ylabel('Feature')
				plt.title(f'{name} subject: Top {top_n} Features for subject {sub} by ANOVA F-score')
				plt.gca().invert_yaxis()  # highest on top
				plt.tight_layout()
				plt.show()

				display(df_cov.head(10))
				subjects_shown_count+=1

	# --- Summarize across subjects ---
	for name in top_features[y_col].keys():
		all_feats = [feat for sublist in top_features[y_col][name].values() for feat in sublist]
		counts = Counter(all_feats)
		top_common = pd.DataFrame(counts.most_common(15), columns=['feature', 'count'])
		
		# --- Visualize Mean Scores Across Subjects ---
		if show_means:
			fig = px.bar(top_common, x='feature', y='count', title=f'Top 15 sig. features to {y_col}  for {name}')
			fig.show()

		print(f'Adding to {name} for {y_col}: {top_common['feature'].to_list()}')
		anova_features[y_col][name] = top_common['feature'].to_list()
	


# %%
print(len(top_features['phq2_sum']['v1_day'].keys()))
print(len(top_features['phq9_sum']['v1_day'].keys()))

print(len(top_features['phq2_sum']['v2_day'].keys()))
print(len(top_features['phq9_sum']['v2_day'].keys()))

print(len(top_features['phq2_sum']['v1_week'].keys()))
print(len(top_features['phq9_sum']['v1_week'].keys()))

print(len(top_features['phq2_sum']['v2_week'].keys()))
print(len(top_features['phq9_sum']['v2_week'].keys()))


# %% [markdown]
# my pipeline: 
# 1. remove outliers
# 4. apply log transform to skewed or wide-tailed data
# 5. remove too-skewed data
# 6. Impute
# 7. Scale
# 8. Drop columns with too few unique values or too little variance

# %% [markdown]
# # Trying to predict each person based on all features, predicting PHQ2 score

# %%
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance
import seaborn as sns

max_iter_list = [50, 100, 200]
colors = sns.color_palette("colorblind")
target = 'phq2_sum'

for name in ['v1_day','v2_day','v1_week','v2_week']:
	print(name)
	df=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_imputed.csv'))
	clean_df = df.dropna(subset=target)
	X = clean_df.drop(columns=[col for col in clean_df.columns if 'phq' in col or col in id_columns or 'Unnamed' in col])
	X = X.select_dtypes(include=('int64', 'float64'))
	y = clean_df[target]
	print(f'Length of DF originally: {len(df)} \nAnd after dropping NA target columns: {len(clean_df)}')
	
	
	splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
	train_idx, test_idx = next(splitter.split(X, y, groups=clean_df['num_id']))

	X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
	y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


	print(f"Training sample size: {X_train.shape[0]}")
	print(f"Test sample size: {X_test.shape[0]}")
	print(f"Number of features: {X_train.shape[1]}")

	fig, ax = plt.subplots(figsize=(10, 5))

	# baseline (actual average PHQ2)
	df_actual = df.loc[X_test.index].copy()
	avg_actual = df_actual.groupby("week")[target].mean()
	avg_actual.plot(color=colors[0], label="Actual average", linewidth=2, ax=ax)

	for idx, max_iter in enumerate(max_iter_list):
		hgbt = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
		hgbt.fit(X_train, y_train)
		y_pred = hgbt.predict(X_test)

		df_pred = df.loc[X_test.index].copy()
		df_pred["y_pred"] = y_pred

		avg_pred = df_pred.groupby("week")["y_pred"].mean()
		avg_pred.plot(
			color=colors[idx + 1],
			label=f"Predicted (max_iter={max_iter})",
			linewidth=2,
			ax=ax,
		)

		if max_iter == max(max_iter_list):
			result = permutation_importance(
				hgbt, X_test, y_test, 
				n_repeats=10, 
				random_state=42,
				n_jobs=-1
			)
			feat_importance_df = pd.DataFrame({
				'feature': X_test.columns,
				'importance_mean': result.importances_mean,
				'importance_std': result.importances_std
			}).sort_values('importance_mean', ascending=False)

			save_path = os.path.join(brighten_dir, f'{name}_feature_importances_maxIter{max_iter}.csv')
			feat_importance_df.to_csv(save_path, index=False)
			display(feat_importance_df.head(8))


	ax.set(
		title="Predicted vs Actual Average PHQ-2 Over Time",
		xlabel="Date",
		ylabel="PHQ-2 (average)",
	)
	ax.legend()
	plt.show()
	print(f'Using features: {X.columns.to_list()}')


# %% [markdown]
# # Trying only using only strongly correlated features

# %%
max_iter_list = [50, 100, 200]
colors = sns.color_palette("colorblind")

strong_corrs = {}
y_col = 'phq2_sum'
similar_y_col = 'phq'
for name in ['v1_day','v2_day','v1_week','v2_week']:
	print(name)
	df=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_transformed.csv'))
	df = df.dropna(subset=y_col)
	corr_df = df.corr(numeric_only=True)[y_col].abs().sort_values(ascending=False)
	corr_df = pd.DataFrame(corr_df).reset_index(names=['column'])
	corr_df[y_col] = pd.to_numeric(corr_df[y_col])
	strong_corr = corr_df.where(corr_df[y_col] > 0.1).dropna(how='all')
	strong_corrs[name] = strong_corr['column'].to_list()
	X_cols = [col for col in strong_corrs[name] if similar_y_col not in col and 'Unnamed' not in col]
	X = df[['num_id']+X_cols]
	y = df[y_col]
	print(f'Length of DF originally: {len(df)} \nAnd after dropping NA target columns: {len(df)}')
	
	
	splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
	train_idx, test_idx = next(splitter.split(X, y, groups=df['num_id']))

	X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
	y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


	print(f"Training sample size: {X_train.shape[0]}")
	print(f"Test sample size: {X_test.shape[0]}")
	print(f"Number of features: {X_train.shape[1]}")

	fig, ax = plt.subplots(figsize=(10, 5))

	# baseline (actual average PHQ2)
	df_actual = df.loc[X_test.index].copy()
	avg_actual = df_actual.groupby("week")[y_col].mean()
	avg_actual.plot(color=colors[0], label="Actual average", linewidth=2, ax=ax)

	for idx, max_iter in enumerate(max_iter_list):
		hgbt = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
		hgbt.fit(X_train, y_train)
		y_pred = hgbt.predict(X_test)

		df_pred = df.loc[X_test.index].copy()
		df_pred["y_pred"] = y_pred

		avg_pred = df_pred.groupby("week")["y_pred"].mean()
		avg_pred.plot(
			color=colors[idx + 1],
			label=f"Predicted (max_iter={max_iter})",
			linewidth=2,
			ax=ax,
		)

		
		if max_iter == max(max_iter_list):
			result = permutation_importance(
				hgbt, X_test, y_test, 
				n_repeats=10, 
				random_state=42,
				n_jobs=-1
			)
			feat_importance_df = pd.DataFrame({
				'feature': X_test.columns,
				'importance_mean': result.importances_mean,
				'importance_std': result.importances_std
			}).sort_values('importance_mean', ascending=False)

			save_path = os.path.join(brighten_dir, f'{name}_feature_importances_maxIter{max_iter}.csv')
			feat_importance_df.to_csv(save_path, index=False)
			display(feat_importance_df.head(6))




	ax.set(
		title="Predicted vs Actual Average PHQ-2 Over Weeks",
		xlabel="Week",
		ylabel="PHQ-2 (average)",
	)
	ax.legend()
	plt.show()

# %% [markdown]
# # Try doing by-subject with top correlated columns

# %%
max_iter_list = [100]
colors = sns.color_palette("colorblind")

test_perc = 0.2

strong_corrs = {}
y_col = 'phq2_sum'
similar_y_col = 'phq'
for name in ['v1_day','v2_day']:
	sub_count=0
	df=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_transformed.csv'))
	for sub, sub_df in df.groupby('num_id'):
		if sub=='1735.0' or sub==1735.0:
			continue
		sub_df_clean = sub_df.dropna(subset=[y_col])
		sub_df_clean = sub_df_clean.sort_values('day', ascending=True)
		days = len(sub_df_clean)
		if days > 20:
			sub_count+=1
			corr_df = sub_df_clean.corr(numeric_only=True)[y_col].abs().sort_values(ascending=False)
			corr_df = pd.DataFrame(corr_df).reset_index(names=['column'])
			corr_df[y_col] = pd.to_numeric(corr_df[y_col])
			strong_corr = corr_df.where(corr_df[y_col] > 0.1).dropna(how='all')
			clean_corr = [col for col in strong_corr['column'].to_list() if col not in id_columns and 'Unnamed' not in col and 'indicator' not in col and col not in ['week','month','day']]
			strong_corrs[sub] = clean_corr
			X_cols = [col for col in strong_corrs[sub] if similar_y_col not in col]
			X = sub_df_clean[X_cols[:10]]

			y = sub_df_clean[y_col]
		
			test_num = int(days*test_perc)
			train_num = days - test_num
			X_train = X[:train_num]
			X_test = X[train_num:]
			y_train = y[:train_num]
			y_test = y[train_num:]

			train_days_for_graphing = sub_df_clean['day'][:train_num]
			test_days_for_graphing = sub_df_clean['day'][train_num:]



			# build your actual_df as before
			actual_df = pd.DataFrame({
				'day': sub_df_clean['day'],
				y_col: y
			})
			
			if sub_count<5:
				fig, ax = plt.subplots(figsize=(10, 5))
				actual_df.plot(
					x='day',
					y=y_col,
					color=colors[0],
					label=f"Actual Daily PHQ-2 over {days} Days",
					linewidth=2,
					ax=ax
				)

			for idx, max_iter in enumerate(max_iter_list):
				hgbt = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
				hgbt.fit(X_train, y_train)
				y_pred = hgbt.predict(X_test)

				pred_df = pd.DataFrame({
					'day': test_days_for_graphing,
					y_col : y_pred
				})

				if sub_count<5:
					pred_df.plot(
						x = 'day',
						y = y_col,
						color=colors[idx + 1],
						label=f"Predicted (max_iter={max_iter}) for the last {test_num} days",
						linewidth=2,
						ax=ax,
					)


				
			
			if sub_count<5:
				ax.set(
					title=f"Sub {sub}, {days} days ({train_num} train), Predicted vs Actual PHQ-2",
					xlabel="Day",
					ylabel="PHQ-2 (Daily)",
				)
				print(f'For {sub}, X-cols: {X.columns.to_list()}')
				ax.legend()
				plt.show()

				
			


# %% [markdown]
# # Try doing by-subject top-10 ANOVA

# %%
import seaborn as sns
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

max_iter_list = [100]
colors = sns.color_palette("colorblind")


test_perc = 0.2

strong_corrs = {}
y_col = 'phq2_sum'
similar_y_col = 'phq'
for name in ['v1_day','v2_day','v1_week','v2_week']:
	sub_count=0
	df=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))
	for sub, sub_df in df.groupby('num_id'):
		if sub=='1735.0' or sub==1735.0:
			continue
		
		try:
			top_features_sub = top_features[y_col][name][sub]
		except Exception as e:
			pass

		sub_df_clean = sub_df.dropna(subset=[y_col])
		sub_df_clean = sub_df_clean.sort_values('day', ascending=True)
		days = len(sub_df_clean)
		if days > 20:
			sub_count+=1
			X = sub_df_clean[top_features_sub]
			y = sub_df_clean[y_col]
		
			test_num = int(days*test_perc)
			train_num = days - test_num
			X_train = X[:train_num]
			X_test = X[train_num:]
			y_train = y[:train_num]
			y_test = y[train_num:]

			train_days_for_graphing = sub_df_clean['day'][:train_num]
			test_days_for_graphing = sub_df_clean['day'][train_num:]



			# build your actual_df as before
			actual_df = pd.DataFrame({
				'day': sub_df_clean['day'],
				y_col: y
			})
			
			if sub_count<5:
				fig, ax = plt.subplots(figsize=(10, 5))
				actual_df.plot(
					x='day',
					y=y_col,
					color=colors[0],
					label=f"Actual Daily PHQ-2 over {days} Days",
					linewidth=2,
					ax=ax
				)

			for idx, max_iter in enumerate(max_iter_list):
				hgbt = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
				hgbt.fit(X_train, y_train)
				y_pred = hgbt.predict(X_test)

				pred_df = pd.DataFrame({
					'day': test_days_for_graphing,
					y_col : y_pred
				})

				if sub_count<5:
					pred_df.plot(
						x = 'day',
						y = y_col,
						color=colors[idx + 1],
						label=f"Predicted (max_iter={max_iter}) for the last {test_num} days",
						linewidth=2,
						ax=ax,
					)


				
			
			if sub_count<5:
				ax.set(
					title=f"Sub {sub}, {days} days ({train_num} train), Predicted vs Actual PHQ-2",
					xlabel="Day",
					ylabel="PHQ-2 (Daily)",
				)
				print(f'For {sub}, X-cols: {X.columns.to_list()}')
				ax.legend()
				plt.show()

				
			


# %% [markdown]
# # By subject using only top 15 ANOVA all-sub features for PHQ2

# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

max_iter_list = [100]
colors = sns.color_palette("colorblind")

test_perc = 0.2
r2_from_model = []

strong_corrs = {}
y_col = 'phq2_sum'
similar_y_col = 'phq'
for name in ['v1_day','v2_day','v1_week','v2_week']:
	print(f'################### {name} ###################')
	sub_count=0
	df=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))
	X_choices = anova_features[y_col][name]
	X_cols = list(set([col for col in X_choices if similar_y_col not in col and 'Unnamed' not in col and col not in id_columns and 'indicator' not in col]))
	X_cols_present = [col for col in df.columns if any(item in col for item in X_cols)]
	print(f'Using cols: {X_cols_present}')
	for sub, sub_df in df.groupby('num_id'):
		sub_count+=1
		sub_df_clean = sub_df.dropna(subset=[y_col])
		sub_df_clean = sub_df_clean.sort_values(by='day', ascending=True)
		days = len(sub_df_clean)
		if days > 20:
			X = sub_df_clean[X_cols_present]
			y = sub_df_clean[y_col]
		
			test_num = int(days*test_perc)
			train_num = days - test_num
			X_train = X[:train_num]
			X_test = X[train_num:]
			y_train = y[:train_num]
			y_test = y[train_num:]

			train_days_for_graphing = sub_df_clean['day'][:train_num]
			test_days_for_graphing = sub_df_clean['day'][train_num:]


			# build your actual_df as before
			actual_df = pd.DataFrame({
				'day': sub_df_clean['day'],
				y_col: y
			})
			

			
			avg = y_train.mean()
			avg_df = pd.DataFrame({
					'day': sub_df_clean['day'],
					y_col : [avg]*days
				})
			
			hgbt = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
			hgbt.fit(X_train, y_train)
			y_pred = hgbt.predict(X_test)

			pred_df = pd.DataFrame({
				'day': test_days_for_graphing,
				y_col : y_pred
			})
			
			r2_actual_vs_model = r2_score(y_test, pred_df[y_col])
			r2_actual_vs_avg = r2_score(y_test, avg_df[y_col][:test_num])
			
			r2_from_model.append([sub, r2_actual_vs_model, r2_actual_vs_avg, r2_actual_vs_model-r2_actual_vs_avg])

			if sub_count<5:
				fig, ax = plt.subplots(figsize=(10, 5))


				actual_df.plot(
					x='day',
					y=y_col,
					color=colors[0],
					label=f"Actual {y_col} over {days} Days",
					linewidth=2,
					ax=ax
				)

				avg_df.plot(
					x='day',
					y=y_col,
					color=colors[1],
					label=f"Average {y_col} over first {train_num} Days",
					linewidth=2,
					ax=ax
				)

				pred_df.plot(
					x = 'day',
					y = y_col,
					color=colors[idx + 2],
					label=f"Predicted (max_iter={max_iter}) for the last {test_num} days",
					linewidth=2,
					ax=ax,
				)

				ax.set(
					title=f"Sub {sub}, {days} days ({train_num} train), Predicted vs Actual {y_col}\nR2 of model prediction: {round(r2_actual_vs_model,2)}\nR2 of average prediction: {round(r2_actual_vs_avg,2)}",
					xlabel="Day",
					ylabel=f"{y_col})",
				)
				ax.legend()
				plt.show()


	r2_from_model_df = pd.DataFrame(r2_from_model, columns=['sub', 'r2_actual_vs_model', 'r2_actual_vs_avg', 'r2_model_increase_from_avg'])
	r2_from_model_df.to_csv(os.path.join(results_dir, f'{name}_{y_col}_r2_model_increase_from_avg.csv'))
			
		


# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))
max_iter_list = [100]
colors = sns.color_palette("colorblind")

test_perc = 0.2
r2_from_model = []

strong_corrs = {}
y_col = 'phq2_sum'
similar_y_col = 'phq'
for name in ['v1_day','v2_day','v1_week','v2_week']:
	print(f'################### {name} ###################')
	sub_count=0
	df=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))
	X_choices = anova_features[y_col][name]
	X_cols = list(set([col for col in X_choices if similar_y_col not in col and 'Unnamed' not in col and col not in id_columns and 'indicator' not in col]))
	X_cols_present = [col for col in df.columns if any(item in col for item in X_cols)]
	print(f'Using cols: {X_cols_present}')
	for sub, sub_df in df.groupby('num_id'):
		sub_count+=1
		sub_df_clean = sub_df.dropna(subset=[y_col])
		sub_df_clean = sub_df_clean.sort_values(by='day', ascending=True)
		days = len(sub_df_clean)
		if days > 20:
			X = sub_df_clean[X_cols_present]
			y = sub_df_clean[y_col]
		
			test_num = int(days*test_perc)
			train_num = days - test_num
			X_train = X[:train_num]
			X_test = X[train_num:]
			y_train = y[:train_num]
			y_test = y[train_num:]

			train_days_for_graphing = sub_df_clean['day'][:train_num]
			test_days_for_graphing = sub_df_clean['day'][train_num:]


			# build your actual_df as before
			actual_df = pd.DataFrame({
				'day': sub_df_clean['day'],
				y_col: y
			})
			

			
			avg = y_train.mean()
			avg_df = pd.DataFrame({
					'day': sub_df_clean['day'],
					y_col : [avg]*days
				})
			
			hgbt = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
			hgbt.fit(X_train, y_train)
			y_pred = hgbt.predict(X_test)

			pred_df = pd.DataFrame({
				'day': test_days_for_graphing,
				y_col : y_pred
			})
			
			r2_actual_vs_model = r2_score(y_test, pred_df[y_col])
			r2_actual_vs_avg = r2_score(y_test, avg_df[y_col][:test_num])
			
			r2_from_model.append([sub, r2_actual_vs_model, r2_actual_vs_avg, r2_actual_vs_model-r2_actual_vs_avg])

			if sub_count<5:
				fig, ax = plt.subplots(figsize=(10, 5))


				actual_df.plot(
					x='day',
					y=y_col,
					color=colors[0],
					label=f"Actual {y_col} over {days} Days",
					linewidth=2,
					ax=ax
				)

				avg_df.plot(
					x='day',
					y=y_col,
					color=colors[1],
					label=f"Average {y_col} over first {train_num} Days",
					linewidth=2,
					ax=ax
				)

				pred_df.plot(
					x = 'day',
					y = y_col,
					color=colors[idx + 2],
					label=f"Predicted (max_iter={max_iter}) for the last {test_num} days",
					linewidth=2,
					ax=ax,
				)

				ax.set(
					title=f"Sub {sub}, {days} days ({train_num} train), Predicted vs Actual {y_col}\nR2 of model prediction: {round(r2_actual_vs_model,2)}\nR2 of average prediction: {round(r2_actual_vs_avg,2)}",
					xlabel="Day",
					ylabel=f"{y_col})",
				)
				ax.legend()
				plt.show()


	r2_from_model_df = pd.DataFrame(r2_from_model, columns=['sub', 'r2_actual_vs_model', 'r2_actual_vs_avg', 'r2_model_increase_from_avg'])
	r2_from_model_df.to_csv(os.path.join(results_dir, f'{name}_{y_col}_r2_model_increase_from_avg.csv'))
			
		


# %% [markdown]
# # PHQ9 Using top 15

# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

max_iter_list = [100]
colors = sns.color_palette("colorblind")

test_perc = 0.2
r2_from_model = []

strong_corrs = {}
y_col = 'phq9_sum'
similar_y_col = 'phq9'
for name in ['v1_week','v2_week']:
	sub_count=0
	df=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))
	X_choices = anova_features[y_col][name]
	X_cols = list(set([col for col in X_choices if similar_y_col not in col and 'Unnamed' not in col and col not in id_columns and 'indicator' not in col and col not in ['week','month','day']]))
	X_cols_present = [col for col in df.columns if any(item in col for item in X_cols)]
	print(f'Using cols: {X_cols_present}')
	for sub, sub_df in df.groupby('num_id'):
		sub_count+=1
		sub_df_clean = sub_df.dropna(subset=[y_col])
		sub_df_clean = sub_df_clean.sort_values(by='week', ascending=True)
		days = len(sub_df_clean)
		if days > 20:
			X = sub_df_clean[X_cols_present]
			y = sub_df_clean[y_col]
		
			test_num = int(days*test_perc)
			train_num = days - test_num
			X_train = X[:train_num]
			X_test = X[train_num:]
			y_train = y[:train_num]
			y_test = y[train_num:]

			test_days_for_graphing = sub_df_clean['week'][train_num:]

			# build your actual_df as before
			actual_df = pd.DataFrame({
				'week': sub_df_clean['week'].copy(),
				y_col: y
			})
			

			
			avg = y_train.mean()
			avg_df = pd.DataFrame({
					'week': sub_df_clean['week'].copy(),
					y_col : [avg]*days
				})
			
			hgbt = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
			hgbt.fit(X_train, y_train)
			y_pred = hgbt.predict(X_test)
			

			pred_df = pd.DataFrame({
				'week': test_days_for_graphing,
				y_col : y_pred
			})
			
			r2_actual_vs_model = r2_score(y_test, pred_df[y_col])
			r2_actual_vs_avg = r2_score(y_test, avg_df[y_col][:test_num])
			
			r2_from_model.append([sub, r2_actual_vs_model, r2_actual_vs_avg, r2_actual_vs_model-r2_actual_vs_avg])

			if sub_count<5:
				display(X_test)
				print(y_pred)
				fig, ax = plt.subplots(figsize=(10, 5))


				actual_df.plot(
					x='week',
					y=y_col,
					color=colors[0],
					label=f"Actual {y_col} over {days} Days",
					linewidth=2,
					ax=ax
				)

				avg_df.plot(
					x='week',
					y=y_col,
					color=colors[1],
					label=f"Average {y_col} over first {train_num} Days",
					linewidth=2,
					ax=ax
				)

				pred_df.plot(
					x = 'week',
					y = y_col,
					color=colors[idx + 2],
					label=f"Predicted (max_iter={max_iter}) for the last {test_num} days",
					linewidth=2,
					ax=ax,
				)

				ax.set(
					title=f"Sub {sub}, {days} days ({train_num} train), Predicted vs Actual {y_col}\nR2 of model prediction: {round(r2_actual_vs_model,2)}\nR2 of average prediction: {round(r2_actual_vs_avg,2)}",
					xlabel="Week",
					ylabel=f"{y_col})",
				)
				ax.legend()
				plt.show()


	r2_from_model_df = pd.DataFrame(r2_from_model, columns=['sub', 'r2_actual_vs_model', 'r2_actual_vs_avg', 'r2_model_increase_from_avg'])
	r2_from_model_df.to_csv(os.path.join(results_dir, f'{name}_{y_col}_r2_model_increase_from_avg.csv'))
			
		


# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))
max_iter_list = [100]
colors = sns.color_palette("colorblind")

test_perc = 0.2
r2_from_model = []

strong_corrs = {}
y_col = 'phq9_sum'
similar_y_col = 'phq9'
for name in ['v1_week','v2_week']:
	sub_count=0
	df=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))
	X_choices = anova_features[y_col][name]
	X_cols = list(set([col for col in X_choices if similar_y_col not in col and 'Unnamed' not in col and col not in id_columns and 'indicator' not in col and col not in ['week','month','day']]))
	X_cols_present = [col for col in df.columns if any(item in col for item in X_cols)]
	print(f'Using cols: {X_cols_present}')
	for sub, sub_df in df.groupby('num_id'):
		sub_count+=1
		sub_df_clean = sub_df.dropna(subset=[y_col])
		sub_df_clean = sub_df_clean.sort_values(by='week', ascending=True)
		days = len(sub_df_clean)
		if days > 20:
			X = sub_df_clean[X_cols_present]
			y = sub_df_clean[y_col]
		
			test_num = int(days*test_perc)
			train_num = days - test_num
			X_train = X[:train_num]
			X_test = X[train_num:]
			y_train = y[:train_num]
			y_test = y[train_num:]

			test_days_for_graphing = sub_df_clean['week'][train_num:]

			# build your actual_df as before
			actual_df = pd.DataFrame({
				'week': sub_df_clean['week'].copy(),
				y_col: y
			})
			

			
			avg = y_train.mean()
			avg_df = pd.DataFrame({
					'week': sub_df_clean['week'].copy(),
					y_col : [avg]*days
				})
			
			hgbt = HistGradientBoostingRegressor(max_iter=max_iter, random_state=42)
			hgbt.fit(X_train, y_train)
			y_pred = hgbt.predict(X_test)
			

			pred_df = pd.DataFrame({
				'week': test_days_for_graphing,
				y_col : y_pred
			})
			
			r2_actual_vs_model = r2_score(y_test, pred_df[y_col])
			r2_actual_vs_avg = r2_score(y_test, avg_df[y_col][:test_num])
			
			r2_from_model.append([sub, r2_actual_vs_model, r2_actual_vs_avg, r2_actual_vs_model-r2_actual_vs_avg])

			if sub_count<5:
				display(X_test)
				print(y_pred)
				fig, ax = plt.subplots(figsize=(10, 5))


				actual_df.plot(
					x='week',
					y=y_col,
					color=colors[0],
					label=f"Actual {y_col} over {days} Days",
					linewidth=2,
					ax=ax
				)

				avg_df.plot(
					x='week',
					y=y_col,
					color=colors[1],
					label=f"Average {y_col} over first {train_num} Days",
					linewidth=2,
					ax=ax
				)

				pred_df.plot(
					x = 'week',
					y = y_col,
					color=colors[idx + 2],
					label=f"Predicted (max_iter={max_iter}) for the last {test_num} days",
					linewidth=2,
					ax=ax,
				)

				ax.set(
					title=f"Sub {sub}, {days} days ({train_num} train), Predicted vs Actual {y_col}\nR2 of model prediction: {round(r2_actual_vs_model,2)}\nR2 of average prediction: {round(r2_actual_vs_avg,2)}",
					xlabel="Week",
					ylabel=f"{y_col})",
				)
				ax.legend()
				plt.show()


	r2_from_model_df = pd.DataFrame(r2_from_model, columns=['sub', 'r2_actual_vs_model', 'r2_actual_vs_avg', 'r2_model_increase_from_avg'])
	r2_from_model_df.to_csv(os.path.join(results_dir, f'{name}_{y_col}_r2_model_increase_from_avg.csv'))
			
		


# %%
for name in ['v1_day','v2_day','v1_week','v2_week']:
	r2_from_model_df=pd.read_csv(os.path.join(results_dir, f'{name}_r2_model_increase_from_avg.csv'))
	display(r2_from_model_df)
	fig = px.bar(r2_from_model_df, x='sub',y='r2_model_increase_from_avg',barmode='group')
	fig.show()


# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

for name in ['v1_day','v2_day','v1_week','v2_week']:
	r2_from_model_df=pd.read_csv(os.path.join(results_dir, f'{name}_r2_model_increase_from_avg.csv'))
	display(r2_from_model_df)
	fig = px.bar(r2_from_model_df, x='sub',y='r2_model_increase_from_avg',barmode='group')
	fig.show()


# %% [markdown]
# ## trying HistGradientBoosting on my data - all features 
#

# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

y_cols = ['phq9_sum']
models = {
	'HistGradientBoostingRegressor': [HistGradientBoostingRegressor(), HistGradientBoostingClassifier()]
}
model_dict = {}
time_cols = ['month','week','day']

for name in ['v1_day','v2_day','v1_week','v2_week']:
	model_dict[name] = {}
	
	print(f'\n\n####### {name} #######')
	for y_col in y_cols:
		model_dict[name][y_col] = {}
		Xy_data=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))

		Xy=Xy_data.dropna(subset=[y_col])
		Xy=Xy.select_dtypes(include=('int64','float64'))

		baseline_cols_present = matches = [item for item in Xy.columns if any(col in item for col in baseline_cols)]
		y_cols_present = [item for item in Xy.columns if any(col in item for col in y_cols) or 'phq9' in item]
		drop_cols = [item for item in Xy.columns if 'Unnamed' in item or '_nan' in item or '_indicator' in item or 'season' in item or 'dt' in item or 'day' in item or 'week' in item or 'month' in item or 'date' in item]

		for time in ['baseline','8wks','both']:
			print(f'\n{time.upper()}')
			model_dict[name][y_col][time] = {}
			if 'baseline' in time:
				X=Xy[['num_id']+time_cols+baseline_cols_present].copy()
			elif '8wks' in time:
				X = Xy.drop(columns=[col for col in baseline_cols_present+y_cols_present+drop_cols if col not in id_columns]).copy()
			else:
				X=Xy.drop(columns=[col for col in y_cols_present+drop_cols if col not in id_columns])
			
			y=Xy[y_col]
			print('Columns in X:',X.columns.to_list())

			# Inputs
			groups = Xy['num_id']
			gkf = GroupKFold(n_splits=5)
			count=0
			print(f'X has {X.shape[0]} observations, {X.shape[1]} features')


			# print('models...')
			for model_name in models:
				model_dict[name][y_col][time][model_name] = {}
				model_dict[name][y_col][time][model_name]['model']=[]
				model_dict[name][y_col][time][model_name]['X_test']=[]
				model_dict[name][y_col][time][model_name]['r2_scores']=[]
				model_dict[name][y_col][time][model_name]['mae_scores']=[]
				model_dict[name][y_col][time][model_name]['y_pred']=[]
				model_dict[name][y_col][time][model_name]['y_test']=[]
				model_dict[name][y_col][time][model_name]['y_val']=[]
				model_dict[name][y_col][time][model_name]['y_valpred']=[]

				for train_idx, test_idx in gkf.split(X, y, groups):
					X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
					y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
					model_dict[name][y_col][time][model_name]['y_test'].append(y_test)
					model_dict[name][y_col][time][model_name]['X_test'].append(X_test)

					model = models[model_name][0]
					if y_col in ['change_cat','change_binary']:
						model= models[model_name][1]
					
					X_train_for_model = X_train.drop(columns=[col for col in time_cols+id_columns if col in X_train.columns and col != 'num_id'])
					if count==0:
						print(f'First CV\n{len(X_train)} observations in X_train, {len(X_test)} in X_test. {X_train['num_id'].nunique()} train subs, {X_test['num_id'].nunique()} test subs')
						print(f'Features: {X_train_for_model.columns.to_list()}')
						print(f'Train Subjects: {set(list(X_train['num_id']))}')
						print(f'Train Subjects: {set(list(X_test['num_id']))}')
						if len(set(list(X_train['num_id'])+list(X_test['num_id']))) == len(set(list(X_train['num_id']))) + len(set(list(X_test['num_id']))):
							print(f'All subjects in each set are unique')
						else:
							print(f"Overlapping subjects")
						count=1


					model.fit(X_train_for_model, y_train)
					assert list(model.feature_names_in_) == list(X_train_for_model.columns), f"Feature mismatch before saving model for {time}"
					model_dict[name][y_col][time][model_name]['model'].append(model)
					
					X_test_for_model = X_test.drop(columns=[col for col in time_cols+id_columns if col in X_train.columns and col != 'num_id'])
					y_pred = model.predict(X_test_for_model)
					model_dict[name][y_col][time][model_name]['y_pred'].append(y_pred)

					

					r2 = r2_score(y_test, y_pred)
					model_dict[name][y_col][time][model_name]['r2_scores'].append(r2)

					mae = mean_absolute_error(y_test, y_pred)
					model_dict[name][y_col][time][model_name]['mae_scores'].append(mae)
					

				print(f"{y_col} from {time} -- mean R²: {round(np.mean(model_dict[name][y_col][time][model_name]['r2_scores']), 4)}, mean MAE: {round(np.mean(model_dict[name][y_col][time][model_name]['mae_scores']), 4)} of scores [{round(y.min(), 1)}-{round(y.max(), 1)}]")
				

# %% [markdown]
# ## Histgradient predicting PHQ9_sum but with only top 15 all-subject ANOVA features

# %%

# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

y_cols = ['phq9_sum']
models = {
	'HistGradientBoostingRegressor': [HistGradientBoostingRegressor(), HistGradientBoostingClassifier()]
}
time = 'anova_15'


for name in ['v1_week','v2_week']:
	print(f'\n\n####### {name} #######')
	for y_col in y_cols:
		model_dict[name][y_col][time] = {}
		Xy=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))

		Xy=Xy.dropna(subset=[y_col])
		X_choices = anova_features[y_col][name]
		X = Xy[[col for col in id_columns+X_choices if col in Xy.columns]]
		print('Columns in X:',X.columns.to_list())
		y = Xy[y_col]

		# Inputs
		groups = Xy['num_id']
		gkf = GroupKFold(n_splits=5)
		count=0
		print(f'X has {X.shape[0]} observations, {X.shape[1]} features')


		# print('models...')
		for model_name in models:
			model_dict[name][y_col][time][model_name] = {}
			model_dict[name][y_col][time][model_name]['model']=[]
			model_dict[name][y_col][time][model_name]['X_test']=[]
			model_dict[name][y_col][time][model_name]['r2_scores']=[]
			model_dict[name][y_col][time][model_name]['mae_scores']=[]
			model_dict[name][y_col][time][model_name]['y_pred']=[]
			model_dict[name][y_col][time][model_name]['y_test']=[]
			model_dict[name][y_col][time][model_name]['y_val']=[]
			model_dict[name][y_col][time][model_name]['y_valpred']=[]
			for train_idx, test_idx in gkf.split(X, y, groups):
				X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
				y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
				model_dict[name][y_col][time][model_name]['y_test'].append(y_test)
				model_dict[name][y_col][time][model_name]['X_test'].append(X_test)

				model = models[model_name][0]
				if y_col in ['change_cat','change_binary']:
					model= models[model_name][1]
				
				X_train_for_model = X_train.drop(columns=[col for col in id_columns if col in X_train.columns])
				if count==0:
					print(f'First CV\n{len(X_train)} observations in X_train, {len(X_test)} in X_test. {X_train['num_id'].nunique()} train subs, {X_test['num_id'].nunique()} test subs')
					print(f'Features: {X_train_for_model.columns.to_list()}')
					print(f'Train Subjects: {set(list(X_train['num_id']))}')
					print(f'Train Subjects: {set(list(X_test['num_id']))}')
					if len(set(list(X_train['num_id'])+list(X_test['num_id']))) == len(set(list(X_train['num_id']))) + len(set(list(X_test['num_id']))):
						print(f'All subjects in each set are unique')
					else:
						print(f"Overlapping subjects")
					count=1


				model.fit(X_train_for_model, y_train)
				assert list(model.feature_names_in_) == list(X_train_for_model.columns), f"Feature mismatch before saving model for {time}"
				model_dict[name][y_col][time][model_name]['model'].append(model)
				
				X_test_for_model = X_test.drop(columns=[col for col in time_cols+id_columns if col in X_test.columns])
				y_pred = model.predict(X_test_for_model)
				model_dict[name][y_col][time][model_name]['y_pred'].append(y_pred)

				

				r2 = r2_score(y_test, y_pred)
				model_dict[name][y_col][time][model_name]['r2_scores'].append(r2)

				mae = mean_absolute_error(y_test, y_pred)
				model_dict[name][y_col][time][model_name]['mae_scores'].append(mae)
				

				print(f"{y_col} from {time} -- mean R²: {round(np.mean(model_dict[name][y_col][time][model_name]['r2_scores']), 4)}, mean MAE: {round(np.mean(model_dict[name][y_col][time][model_name]['mae_scores']), 4)} of scores [{round(y.min(), 1)}-{round(y.max(), 1)}]")
				





# %%

# %% [markdown]
# # SHAP for the above predictive models

# %%
# SHAP Printing -- do after above, can't run without first making results {}
import gc 

for name in ['v1_week', 'v2_week']: #4
	for y_col in ['phq9_sum']: #2
		for time in ['8wks','both','top 15 ANOVA']: #3
			for model_name in models: #only one atm 
				mean_r2 = round(np.mean(model_dict[name][y_col][time][model_name]['r2_scores']), 3)
				print(name, time, model_name, f'Mean R2: {mean_r2}')
				i=0 #change based on which CV you want to see
				df = model_dict[name][y_col][time][model_name]['X_test'][i]
				df_model = df[[col for col in df.columns if col not in id_columns]]
				explainer = shap.Explainer(model_dict[name][y_col][time][model_name]['model'][i], approximate=True)
				shap_values = explainer(df_model, check_additivity=False)
				shap.plots.bar(shap_values)
				del explainer, shap_values
				gc.collect()


# %%
for name in ['v1_week', 'v2_week']: #4
	for y_col in ['phq9_sum']: #2
		for time in ['8wks','both','anova_15']: #3
			for model_name in models: #only one atm 

				X_test = model_dict[name][y_col][time][model_name]['X_test'][0]
				assert 'num_id' in X_test.columns.to_list()
				assert 'day' in X_test.columns.to_list()



# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

import gc 
num_to_show=2

for name in ['v1_week', 'v2_week']: #4
	for y_col in ['phq9_sum']: #2
		for time in ['8wks','both','anova_15']: #3
			for model_name in models: #only one atm 
				num_shown=0
				mean_r2 = round(np.mean(model_dict[name][y_col][time][model_name]['r2_scores']), 3)
				print(name, time, model_name, f'Mean R2: {mean_r2}')
				i=0 #change based on which CV you want to see
				# df = model_dict[name][y_col][time][model_name]['X_test'][i]
				# df_model = df[[col for col in df.columns if col not in id_columns]]
				# explainer = shap.Explainer(model_dict[name][y_col][time][model_name]['model'][i], approximate=True)
				# shap_values = explainer(df_model, check_additivity=False)
				# shap.plots.bar(shap_values)
				# del explainer, shap_values
				# gc.collect()  

				X_test = model_dict[name][y_col][time][model_name]['X_test'][i]
				y_pred = model_dict[name][y_col][time][model_name]['y_pred'][i]
				y_test = model_dict[name][y_col][time][model_name]['y_test'][i]

				Xy = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))

				y_df = pd.DataFrame({
					f'{y_col}_pred': y_pred, 
					f'{y_col}_actual': y_test.values
					}, index=y_test.index)

				meta = Xy.loc[y_test.index, ['num_id', 'day', 'week']]
				X_test_no_ids = X_test.drop(columns=[col for col in id_columns if col in X_test.columns])
				X_test_no_ids = X_test_no_ids.set_index(y_test.index) if not X_test_no_ids.index.equals(y_test.index) else X_test_no_ids

				df = pd.concat([meta, X_test_no_ids, y_df], axis=1)

				# verify no duplicate columns
				assert not df.columns.duplicated().any(), f"Duplicate cols: {df.columns[df.columns.duplicated()].tolist()}"

				for sub, sub_df in df.groupby('num_id'):
					if len(sub_df.dropna(subset=f'{y_col}_pred'))==0:
						continue
					
					if num_shown<=num_to_show:
						sub_df = sub_df.sort_values(by='day')

						fig = px.line(sub_df, x='day', y=[f'{y_col}_pred', f'{y_col}_actual'], 
						title=f'For sub {sub}, using {X_test.shape[1]} {time} features,  pred. vs. actual {y_col} score -- Mean R2 = {mean_r2} over 5 CVs')

						fig.show()
						num_shown+=1



			

# %% [markdown]
# # NO PHQ2 or other related variables (SDS, stress, mood)
#

# %%

y_cols = ['phq9_sum']
drop_survey_cols = ['phq2_1','phq2_2','phq2_sum','phq2_bin','sds_1','sds_2','sds_3','stress','support','mhs_1','mhs_2','mhs_3','mhs_4','mhs_5'] #sum up the MHS vars
end_cols = ['6wks_depressed_binary', 'depression_change_bin']
drop_cols_base = y_cols+drop_survey_cols+end_cols
drop_cols_with_this_present = ['phq9','Unnamed','_nan','_indicator','race','season','cohort'] #take out race because it adds 8 vars

time_cols = ['month','week','day']

models = {
	'HistGradientBoostingRegressor': [HistGradientBoostingRegressor(), HistGradientBoostingClassifier()]
}


time = 'only_sensor'

for name in ['v1_week','v2_week']:
	
	print(f'\n\n####### {name} #######')
	for y_col in y_cols:
		
		Xy=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))

		Xy=Xy.dropna(subset=[y_col])
		Xy=Xy.select_dtypes(include=('int64','float64'))
		print(f"Cols in Xy: {Xy.columns.to_list()}")
		y=Xy[y_col]
		baseline_cols_present = matches = [item for item in Xy.columns if any(col in item for col in baseline_cols)]
		drop_cols = drop_cols_base.copy()
		drop_additional_cols = [item for item in Xy.columns if any(col in item for col in drop_cols_with_this_present)]
		drop_cols += drop_additional_cols		


		print(f'\n{time.upper()}')
		model_dict[name][y_col][time] = {}
		if 'baseline' in time:
			X1=Xy[['num_id']+time_cols+baseline_cols_present].copy()
			
		elif '8wks' in time:
			drop_cols = [col for col in drop_cols if col in Xy.columns]
			X1 = Xy.drop(columns=[col for col in baseline_cols_present+drop_cols if col not in id_columns] ).copy()
		else:
			drop_cols = [col for col in drop_cols if col in Xy.columns]
			X1=Xy.drop(columns=[col for col in drop_cols if col not in id_columns])
		
		# Inputs
		groups = Xy['num_id']
		gkf = GroupKFold(n_splits=5)
		count=0
		print(f'X has {X1.shape[0]} observations, {X1.shape[1]} features')
		print(f'Cols in X: {X1.columns.to_list()}')


		# print('models...')
		for model_name in models:
			model_dict[name][y_col][time][model_name] = {}
			model_dict[name][y_col][time][model_name]['model']=[]
			model_dict[name][y_col][time][model_name]['X_test']=[]
			model_dict[name][y_col][time][model_name]['X_test_for_model']=[]
			model_dict[name][y_col][time][model_name]['r2_scores']=[]
			model_dict[name][y_col][time][model_name]['mae_scores']=[]
			model_dict[name][y_col][time][model_name]['y_pred']=[]
			model_dict[name][y_col][time][model_name]['y_test']=[]
			model_dict[name][y_col][time][model_name]['y_val']=[]
			model_dict[name][y_col][time][model_name]['y_valpred']=[]

			
			for train_idx, test_idx in gkf.split(X1, y, groups):
				X_train, X_test = X1.iloc[train_idx], X1.iloc[test_idx]
				y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
				model_dict[name][y_col][time][model_name]['y_test'].append(y_test)
				model_dict[name][y_col][time][model_name]['X_test'].append(X_test)


				model = models[model_name][0]
				if y_col in ['change_cat','change_binary']:
					model= models[model_name][1]
				
				present_time_cols = [col for col in time_cols+id_columns if col in X_train.columns]
				X_train_for_model = X_train.drop(columns=present_time_cols)
				if count==0:
					print(f'cols for {X_train_for_model.columns.to_list()}')
					print(f'First CV\n{len(X_train)} observations in X_train, {len(X_test)} in X_test. {X_train['num_id'].nunique()} train subs, {X_test['num_id'].nunique()} test subs')
					print(f'Features in X_train_for_model: {X_train_for_model.columns.to_list()}')
					print(f'Train Subjects ({len(set(list(X_train['num_id'])))}) {set(list(X_train['num_id']))}')
					print(f'Test Subjects ({len(set(list(X_test['num_id'])))}): {set(list(X_test['num_id']))}')
					if len(set(list(X_train['num_id'])+list(X_test['num_id']))) == len(set(list(X_train['num_id']))) + len(set(list(X_test['num_id']))):
						print(f'All subjects in each set are unique')
					else:
						print(f"Overlapping subjects")
					count=1


				model.fit(X_train_for_model, y_train)
				assert list(model.feature_names_in_) == list(X_train_for_model.columns), f"Feature mismatch before saving model for {time}"
				model_dict[name][y_col][time][model_name]['model'].append(model)
				
				X_test_for_model = X_test.drop(columns=[col for col in time_cols+id_columns if col in X_test.columns])
				model_dict[name][y_col][time][model_name]['X_test_for_model'].append(X_test_for_model)

				y_pred = model.predict(X_test_for_model)

				model_dict[name][y_col][time][model_name]['y_pred'].append(y_pred)

				r2 = r2_score(y_test, y_pred)
				model_dict[name][y_col][time][model_name]['r2_scores'].append(r2)

				mae = mean_absolute_error(y_test, y_pred)
				model_dict[name][y_col][time][model_name]['mae_scores'].append(mae)
				

			print(f"{y_col} from {time} -- mean R²: {round(np.mean(model_dict[name][y_col][time][model_name]['r2_scores']), 4)}, mean MAE: {round(np.mean(model_dict[name][y_col][time][model_name]['mae_scores']), 4)} of scores [{round(y.min(), 1)}-{round(y.max(), 1)}]")
			
			# X_val = val_df[[col for col in val_df.columns if col in X_test]]
			# X_val_for_model = val_df[[col for col in X_val.columns if col in X_test_for_model]]
			# X_val_for_model = X_val_for_model[X_test_for_model.columns]
			# yval_pred = model.predict(X_val_for_model)
			# model_dict[name][y_col][time][model_name]['y_valpred'].append(yval_pred)
			# yval_true = val_df[y_col]
			# model_dict[name][y_col][time][model_name]['y_val'].append(yval_true)
			
			# r2 = r2_score(y_test, y_pred)
			# print(f"FOR VAL:: {y_col} from {time} -- R²: {r2}")







# %%
#where i stopped mon apr 13 1:09am
import gc 

subject_scores = []

for name in ['v1_week','v2_week']: #4
	for y_col in ['phq9_sum']: #2
		for time in ['8wks','both','only_sensor','anova_15']: #3
			for model_name in models: #only one atm 
				mean_r2 = round(np.mean(model_dict[name][y_col][time][model_name]['r2_scores']), 3)
				print(name, time, model_name, f'Mean R2: {mean_r2}')
				i=0 #change based on which CV you want to see
				model = model_dict[name][y_col][time][model_name]['model'][i]
				explainer = shap.Explainer(model_dict[name][y_col][time][model_name]['model'][i], approximate=True)
				test_df = model_dict[name][y_col][time][model_name]['X_test'][i]
				x_test_for_model = test_df.drop(columns=[col for col in test_df.columns if col in id_columns])
				shap_values = explainer(x_test_for_model, check_additivity=False)
				shap.plots.bar(shap_values)
				del explainer, shap_values
				gc.collect()  

				X_test = model_dict[name][y_col][time][model_name]['X_test'][i] # use X_test for the plotting bc it has the day/week variable
				y_pred = model_dict[name][y_col][time][model_name]['y_pred'][i]
				y_test = model_dict[name][y_col][time][model_name]['y_test'][i]

				y_df = pd.DataFrame({f'{y_col}_pred': y_pred, f'{y_col}_actual': y_test})
				df = pd.concat([X_test, y_df], axis=1)

				print(list(model.feature_names_in_), '\n')

				
				for sub, sub_df in df.groupby('num_id'):
					if len(sub_df.dropna(subset=f'{y_col}_pred'))<3:
						continue

					sub_df = sub_df.sort_values(by='day')

					r2_score_sub = r2_score(list(sub_df[f'{y_col}_actual']), list(sub_df[f'{y_col}_pred']))
					pmc_score_sub = pearsonr(list(sub_df[f'{y_col}_actual']), list(sub_df[f'{y_col}_pred']))

					fig = px.line(sub_df, x='day', y=[f'{y_col}_pred', f'{y_col}_actual'], 
					title=f'Sub {sub}, {X_test.shape[1]} {time} feats, pred. vs. actual {y_col} score | R2:{round(r2_score_sub, 2)}, PMC:{round(pmc_score_sub[0], 2)}')

					fig.show()

					subject_scores.append([name, y_col, time, model_name, sub, r2_score_sub, pmc_score_sub[0]])

subject_scores_df = pd.DataFrame(subject_scores, columns=['name', 'y_col', 'time', 'model_name', 'sub', 'r2_score_sub', 'pmc_score_sub'])
subject_scores_df.to_csv(os.path.join(results_dir, 'subject_phq9_preds_from_diff_models.csv'))

# %%

import gc 

subject_scores = []

for name in ['v1_week','v2_week']: #4
	for y_col in ['phq9_sum']: #2
		for time in ['8wks','both']: #3
			for model_name in models: #only one atm 
				mean_r2 = round(np.mean(model_dict[name][y_col][time][model_name]['r2_scores']), 3)
				print(name, type, model_name, f'Mean R2: {mean_r2}')
				i=0 #change based on which CV you want to see
				model = model_dict[name][y_col][time][model_name]['model'][i]
				explainer = shap.Explainer(model_dict[name][y_col][time][model_name]['model'][i], approximate=True)
				test_df = model_dict[name][y_col][time][model_name]['X_test'][i]
				x_test_for_model = test_df.drop(columns=[col for col in test_df.columns if col in id_columns])
				shap_values = explainer(x_test_for_model, check_additivity=False)
				shap.plots.bar(shap_values)
				del explainer, shap_values
				gc.collect()  

				X_test = model_dict[name][y_col][time][model_name]['X_test'][i] # use X_test for the plotting bc it has the day/week variable
				y_pred = model_dict[name][y_col][time][model_name]['y_pred'][i]
				y_test = model_dict[name][y_col][time][model_name]['y_test'][i]

				y_df = pd.DataFrame({f'{y_col}_pred': y_pred, f'{y_col}_actual': y_test})
				df = pd.concat([X_test, y_df], axis=1)

				print(list(model.feature_names_in_), '\n')

				
				for sub, sub_df in df.groupby('num_id'):
					if len(sub_df.dropna(subset=f'{y_col}_pred'))<3:
						continue

					sub_df = sub_df.sort_values(by='day')

					r2_score_sub = r2_score(list(sub_df[f'{y_col}_actual']), list(sub_df[f'{y_col}_pred']))
					pmc_score_sub = pearsonr(list(sub_df[f'{y_col}_actual']), list(sub_df[f'{y_col}_pred']))

					fig = px.line(sub_df, x='day', y=[f'{y_col}_pred', f'{y_col}_actual'], 
					title=f'Sub {sub}, {X_test.shape[1]} {time} feats, pred. vs. actual {y_col} score | R2:{round(r2_score_sub, 2)}, PMC:{round(pmc_score_sub[0], 2)}')

					fig.show()

					subject_scores.append([name, y_col, time, model_name, sub, r2_score_sub, pmc_score_sub[0]])

subject_scores_df = pd.DataFrame(subject_scores, columns=['name', 'y_col', 'time', 'model_name', 'sub', 'r2_score_sub', 'pmc_score_sub'])
subject_scores_df.to_csv(os.path.join(results_dir, 'subject_phq9_preds_from_diff_models.csv'))

# %%
subject_scores_df = pd.read_csv(os.path.join(results_dir, 'subject_phq9_preds_from_diff_models.csv'))
time_map = {'8wks': 'Passive Variables over 8 weeks', 'both': 'Passive Variables over 8 weeks + Baseline clinical and demographic features'}
for time in time_map.keys():
	for metric in ['r2_score_sub','pmc_score_sub']:
		subject_scores_df=subject_scores_df.mask(subject_scores_df[metric]<-1)
		# assuming your DataFrame is called df
		fig = px.bar(
			subject_scores_df[subject_scores_df['time']==time],
			x='sub',
			y=metric,
			title=f'{metric} Score per Subject',
			labels={'sub': 'Subject ID'},
			color='name'
		)

		# Optional: improve layout and readability
		fig.update_layout(
			xaxis={'type': 'category'},  # keeps x-axis discrete even if numeric
			xaxis_tickangle=-45,
			plot_bgcolor='white',
			yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black'),
		)
		
		fig.show()

# %%
subject_scores_df = pd.read_csv(os.path.join(results_dir, 'subject_phq9_preds_from_diff_models.csv'))
time_map = {'8wks': 'Passive Variables over 8 weeks', 'both': 'Passive Variables over 8 weeks + Baseline clinical and demographic features'}
for time in time_map.keys():
	for metric in ['r2_score_sub','pmc_score_sub']:
		subject_scores_df=subject_scores_df.mask(subject_scores_df[metric]<-1)
		# assuming your DataFrame is called df
		fig = px.bar(
			subject_scores_df[subject_scores_df['time']==time],
			x='sub',
			y=metric,
			title=f'{metric} Score per Subject',
			labels={'sub': 'Subject ID'},
			color='name'
		)

		# Optional: improve layout and readability
		fig.update_layout(
			xaxis={'type': 'category'},  # keeps x-axis discrete even if numeric
			xaxis_tickangle=-45,
			plot_bgcolor='white',
			yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black'),
		)
		
		fig.show()

# %%

# %% [markdown]
# # PHQ2 sum predicting

# %%

y_cols = ['phq2_sum']
drop_survey_cols = ['phq2_1','phq2_2','phq2_sum','phq2_bin','sds_1','sds_2','sds_3','stress','support','mhs_1','mhs_2','mhs_3','mhs_4','mhs_5'] #sum up the MHS vars
end_cols = ['6wks_depressed_binary', 'depression_change_bin']
drop_cols_base = y_cols+drop_survey_cols+end_cols
drop_cols_with_this_present = ['phq2','phq9','Unnamed','_nan','_indicator','race','season','cohort','dt'] #take out race because it adds 8 vars

time_cols = ['month','week','day']

models = {
	'HistGradientBoostingRegressor': [HistGradientBoostingRegressor(), HistGradientBoostingClassifier()]
}


times = ['8wks','both']
for name in ['v1_day','v2_day']:
	
	print(f'\n\n####### {name} #######')
	for y_col in y_cols:
		if not y_col in model_dict[name].keys():
			model_dict[name][y_col]={}
		
		Xy=pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_pca.csv'))

		Xy=Xy.dropna(subset=[y_col])
		Xy=Xy.select_dtypes(include=('int64','float64'))
		y=Xy[y_col]
		baseline_cols_present = matches = [item for item in Xy.columns if any(col in item for col in baseline_cols)]
		drop_cols = drop_cols_base.copy()
		drop_additional_cols = [item for item in Xy.columns if any(col in item for col in drop_cols_with_this_present)]
		drop_cols += drop_additional_cols		

		for time in times:
			print(f'\n{time.upper()}')
			model_dict[name][y_col][time] = {}
			if 'baseline' in time:
				X1=Xy[['num_id']+time_cols+baseline_cols_present].copy()
				
			elif '8wks' in time:
				drop_cols = [col for col in drop_cols if col in Xy.columns]
				X1 = Xy.drop(columns=[col for col in baseline_cols_present+drop_cols if col not in id_columns] ).copy()
			else:
				drop_cols = [col for col in drop_cols if col in Xy.columns]
				X1=Xy.drop(columns=[col for col in drop_cols if col not in id_columns])
			
			# Inputs
			groups = Xy['num_id']
			gkf = GroupKFold(n_splits=5)
			count=0
			print(f'X has {X1.shape[0]} observations, {X1.shape[1]} features')
			print(f'Cols in X: {X1.columns.to_list()}')


			# print('models...')
			for model_name in models:
				model_dict[name][y_col][time][model_name] = {}
				model_dict[name][y_col][time][model_name]['model']=[]
				model_dict[name][y_col][time][model_name]['X_test']=[]
				model_dict[name][y_col][time][model_name]['X_test_for_model']=[]
				model_dict[name][y_col][time][model_name]['r2_scores']=[]
				model_dict[name][y_col][time][model_name]['mae_scores']=[]
				model_dict[name][y_col][time][model_name]['y_pred']=[]
				model_dict[name][y_col][time][model_name]['y_test']=[]
				model_dict[name][y_col][time][model_name]['y_val']=[]
				model_dict[name][y_col][time][model_name]['y_valpred']=[]

				
				for train_idx, test_idx in gkf.split(X1, y, groups):
					X_train, X_test = X1.iloc[train_idx], X1.iloc[test_idx]
					y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
					model_dict[name][y_col][time][model_name]['y_test'].append(y_test)
					model_dict[name][y_col][time][model_name]['X_test'].append(X_test)


					model = models[model_name][0]
					if y_col in ['change_cat','change_binary']:
						model= models[model_name][1]
					
					present_time_cols = [col for col in time_cols+id_columns if col in X_train.columns]
					X_train_for_model = X_train.drop(columns=present_time_cols)
					if count==0:
						print(f'Model cols: {X_train_for_model.columns.to_list()}')
						print(f'First CV\n{len(X_train)} observations in X_train, {len(X_test)} in X_test. {X_train['num_id'].nunique()} train subs, {X_test['num_id'].nunique()} test subs')
						print(f'Features in X_train_for_model: {X_train_for_model.columns.to_list()}')
						print(f'Train Subjects ({len(set(list(X_train['num_id'])))}) {set(list(X_train['num_id']))}')
						print(f'Test Subjects ({len(set(list(X_test['num_id'])))}): {set(list(X_test['num_id']))}')
						if len(set(list(X_train['num_id'])+list(X_test['num_id']))) == len(set(list(X_train['num_id']))) + len(set(list(X_test['num_id']))):
							print(f'All subjects in each set are unique')
						else:
							print(f"Overlapping subjects")
						count=1


					model.fit(X_train_for_model, y_train)
					assert list(model.feature_names_in_) == list(X_train_for_model.columns), f"Feature mismatch before saving model for {time}"
					model_dict[name][y_col][time][model_name]['model'].append(model)
					
					X_test_for_model = X_test.drop(columns=[col for col in time_cols+id_columns if col in X_test.columns])
					model_dict[name][y_col][time][model_name]['X_test_for_model'].append(X_test_for_model)

					y_pred = model.predict(X_test_for_model)

					model_dict[name][y_col][time][model_name]['y_pred'].append(y_pred)

					r2 = r2_score(y_test, y_pred)
					model_dict[name][y_col][time][model_name]['r2_scores'].append(r2)

					mae = mean_absolute_error(y_test, y_pred)
					model_dict[name][y_col][time][model_name]['mae_scores'].append(mae)
					

				print(f"{y_col} from {time} -- mean R²: {round(np.mean(model_dict[name][y_col][time][model_name]['r2_scores']), 4)}, mean MAE: {round(np.mean(model_dict[name][y_col][time][model_name]['mae_scores']), 4)} of scores [{round(y.min(), 1)}-{round(y.max(), 1)}]")
				







# %%

import gc 

subject_scores = []

for name in ['v1_day','v2_day']: #4
	for y_col in ['phq2_sum']: #2
		for time in ['8wks','both']: #3
			for model_name in models: #only one atm 
				mean_r2 = round(np.mean(model_dict[name][y_col][time][model_name]['r2_scores']), 3)
				print(name, type, model_name, f'Mean R2: {mean_r2}')
				i=0 #change based on which CV you want to see
				model = model_dict[name][y_col][time][model_name]['model'][i]
				explainer = shap.Explainer(model_dict[name][y_col][time][model_name]['model'][i], approximate=True)
				shap_values = explainer(model_dict[name][y_col][time][model_name]['X_test_for_model'][i], check_additivity=False)
				shap.plots.bar(shap_values)
				del explainer, shap_values
				gc.collect()  

				X_test = model_dict[name][y_col][time][model_name]['X_test'][i] # use X_test for the plotting bc it has the day/week variable
				y_pred = model_dict[name][y_col][time][model_name]['y_pred'][i]
				y_test = model_dict[name][y_col][time][model_name]['y_test'][i]

				y_df = pd.DataFrame({f'{y_col}_pred': y_pred, f'{y_col}_actual': y_test})
				df = pd.concat([X_test, y_df], axis=1)

				print(list(model.feature_names_in_), '\n')

				
				for sub, sub_df in df.groupby('num_id'):
					if len(sub_df.dropna(subset=f'{y_col}_pred'))<3:
						continue

					sub_df = sub_df.sort_values(by='day')

					r2_score_sub = r2_score(list(sub_df[f'{y_col}_actual']), list(sub_df[f'{y_col}_pred']))
					pmc_score_sub = pearsonr(list(sub_df[f'{y_col}_actual']), list(sub_df[f'{y_col}_pred']))

					fig = px.line(sub_df, x='day', y=[f'{y_col}_pred', f'{y_col}_actual'], 
					title=f'Sub {sub}, {X_test.shape[1]} {time} feats, pred. vs. actual {y_col} score | R2:{round(r2_score_sub, 2)}, PMC:{round(pmc_score_sub[0], 2)}')					
					fig.show()

					subject_scores.append([name, y_col, time, model_name, sub, r2_score_sub, pmc_score_sub[0]])

subject_scores_df = pd.DataFrame(subject_scores, columns=['name', 'y_col', 'time', 'model_name', 'sub', 'r2_score_sub', 'pmc_score_sub'])
subject_scores_df.to_csv(os.path.join(results_dir, 'subject_phq2_preds_from_diff_models.csv'))

# %% [markdown]
# # Subject Scores for PHQ2 prediction

# %%

subject_scores_df = pd.read_csv(os.path.join(results_dir, 'subject_phq2_preds_from_diff_models.csv'))
time_map = {'8wks': 'Passive Variables over 8 weeks', 'both': 'Passive Variables over 8 weeks + Baseline clinical and demographic features'}
for time in time_map.keys():
	for metric in ['r2_score_sub','pmc_score_sub']:
		subject_scores_df=subject_scores_df.mask(subject_scores_df[metric]<-1)
		# assuming your DataFrame is called df
		fig = px.bar(
			subject_scores_df[subject_scores_df['time']==time],
			x='sub',
			y=metric,
			title=f'{metric} Score per Subject',
			labels={'sub': 'Subject ID'},
			color='name'
		)

		# Optional: improve layout and readability
		fig.update_layout(
			xaxis={'type': 'category'},  # keeps x-axis discrete even if numeric
			xaxis_tickangle=-45,
			plot_bgcolor='white',
			yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black'),
		)
		
		fig.show()

# %%

# %% [markdown]
# # Modelling using slope, int and mean features with ONLY first 4 weeks
# ## across all subjects
#

# %%

from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import f_classif, SelectKBest
fs = SelectKBest(score_func=f_classif, k=25)

print('Using smartphone sensor data + surveys, over 8 weeks')

y_cols =['phq9_sum_6wks', '6wks_depressed_binary', 'depression_change_bin']
models = {
	'HistGradientBoostingRegressor': [HistGradientBoostingRegressor(), HistGradientBoostingClassifier()]
}

model_dict_slopeint = {}
count=0
for name in ['v1_day', 'v2_day']: 
	Xy = pd.read_csv(os.path.join(brighten_dir, f"{name}_wide_slopeintercept_2wk_outcomes.csv"))
	model_dict_slopeint[name] = {}
	

	print(f'####### {name} #######')
	for y_col in y_cols:
		Xy = Xy.dropna(subset=[y_col])

		model_dict_slopeint[name][y_col] = {}
		for time in ['8wks','both','baseline']:
			print(f'\n\nPREDICTING: {y_col} from {time}')
			if 'baseline' in time:
				usable_cols = baseline_cols
			if '8wks' in time:
				usable_cols = daily_cols_v1+daily_cols_v2+weekly_cols
			elif 'both' in time:
				usable_cols = baseline_cols+daily_cols_v1+daily_cols_v2+weekly_cols

			model_dict_slopeint[name][y_col][time]={}
			
			y=Xy[y_col]
			# columns to drop
			X=Xy[[col for col in Xy.columns if any(item in col for item in usable_cols) and 'phq' not in col]].copy()
			X=X.drop(columns=[col for col in X.columns if 'block3' in col or 'block4' in col or 'block5' in col or 'block6' in col]).copy()	
	
			print(f'X has {X.shape[0]} observations, {X.shape[1]} features')
			
			#display(X)
			non_numeric_cols = X.select_dtypes(include=['object']).columns
			if len(non_numeric_cols) > 0:
				print("Non-numeric columns found:", non_numeric_cols)
				print(X[non_numeric_cols].head())
			
			# Inputs
			groups = Xy['num_id']
			gkf = GroupKFold(n_splits=5)


			for model_name in models:
				model_dict_slopeint[name][y_col][time][model_name] = {}
				model_dict_slopeint[name][y_col][time][model_name]['model']=[]
				model_dict_slopeint[name][y_col][time][model_name]['X_test']=[]
				model_dict_slopeint[name][y_col][time][model_name]['r2_scores']=[]
				model_dict_slopeint[name][y_col][time][model_name]['mae_scores']=[]
				for train_idx, test_idx in gkf.split(X, y, groups):
					X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
					y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

					if X_train.shape[0]==0 or X_train.shape[1]==0 or len(X_train)==0:
						model_dict_slopeint[name][y_col][time][model_name]['model'].append(np.nan)
						model_dict_slopeint[name][y_col][time][model_name]['X_test'].append(np.nan)
						model_dict_slopeint[name][y_col][time][model_name]['r2_scores'].append(np.nan)
						model_dict_slopeint[name][y_col][time][model_name]['mae_scores'].append(np.nan)
						continue

					model = models[model_name][0]
					if y_col in ['start_depressed_binary','6wks_depressed_binary','depression_change_bin','bin_clin']:
						model= models[model_name][1]

					model.fit(X_train, y_train)
					model_dict_slopeint[name][y_col][time][model_name]['model'].append(model)

					y_pred = np.round(model.predict(X_test)).astype(int) #ordinal Y
					model_dict_slopeint[name][y_col][time][model_name]['X_test'].append(X_test)

					r2 = r2_score(y_test, y_pred)
					model_dict_slopeint[name][y_col][time][model_name]['r2_scores'].append(r2)

					mae = mean_absolute_error(y_test, y_pred)
					model_dict_slopeint[name][y_col][time][model_name]['mae_scores'].append(mae)
					

				print(f"TIME {time} -- mean R²: {round(np.mean(model_dict_slopeint[name][y_col][time][model_name]['r2_scores']), 4)}, mean MAE: {round(np.mean(model_dict_slopeint[name][y_col][time][model_name]['mae_scores']), 4)} of {round(y.min(), 1)}-{round(y.max(), 1)}\n")




# %% [markdown]
# # Now show the feature importance for the above
#

# %%
i=0 #change based on which CV you want to see
count=0
top15features = {}
for name, y_dict in model_dict_slopeint.items():
	top15features[name] = {}
	for y_col, type_dict in y_dict.items():
		top15features[name][y_col] = {}
		for time, model_dict in type_dict.items():
			top15features[name][y_col][time]={}
			for model_name, metrics in model_dict.items():
				# mean_r2 = round(np.mean(model_dict_slopeint[name][y_col][time][model_name]['r2_scores']), 3)
				print('name',name, 'y_col',y_col, 'time',time, 'model_name',model_name, 'mean r2', round(np.mean(model_dict_slopeint[name][y_col][time][model_name]['r2_scores']), 4))
		
				model = model_dict_slopeint[name][y_col][time][model_name]['model'][i]
				X_test = model_dict_slopeint[name][y_col][time][model_name]['X_test'][i]
				
				if name == 'v1_day' and y_col == 'start_depressed_binary' and time == '8wks':
					if count<1:
						print(X_test.columns.to_list())
						count=1

				if model is None or X_test is None or len(X_test) == 0:
					continue

				# === Choose the right SHAP explainer dynamically ===
				if y_col in ['start_depressed_binary','6wks_depressed_binary','depression_change_bin']:
					continue
					# fallback to model-agnostic Explainer for Classifier
					explainer = shap.Explainer(model.predict_proba, X_test, feature_names=X_test.columns)
					shap_values = explainer(X_test)

					# SHAP gives one explanation per class → pick positive class (index 1)
					shap_values = shap_values[:, :, 1]
				else:
					# normal tree explainer for Regressor
					explainer = shap.TreeExplainer(model, approximate=True)
					shap_values = explainer(X_test, check_additivity=False)

				# === Plot mean absolute SHAP values per feature ===
				shap_df = pd.DataFrame({
					'columns': X_test.columns,
					'shap_values': np.abs(shap_values.values).mean(axis=0)
				})
				display(shap_df.sort_values('shap_values', ascending=False).head(10))
				top15features[name][y_col][time] = list(shap_df['columns'])[:20]


				shap.plots.bar(shap_values, max_display=10)

				del explainer, shap_values
				gc.collect()

				shap_df = shap_df.sort_values(by='shap_values', ascending=False) #sort from highest to lowest
				

# %% [markdown]
# # Using only the top 10 features from feature importance 
# ### Still slope/int and block1+2 to predict week 6

# %%


model_dict_slopeint_top10 = {}
y_cols =['phq9_sum_6wks', '6wks_depressed_binary', 'depression_change_bin']

models = {
	'HistGradientBoostingRegressor': [HistGradientBoostingRegressor(), HistGradientBoostingClassifier()]
}

for name in ['v1_day','v2_day']: 
	df = pd.read_csv(os.path.join(brighten_dir, f"{name}_wide_slopeintercept_2wk_outcomes.csv"))
	model_dict_slopeint_top10[name] = {}
	

	print(f'####### {name} #######')
	for y_col in y_cols:
		model_dict_slopeint_top10[name][y_col]={}
		df_clean = df.dropna(subset=[y_col])
		model_dict_slopeint_top10[name][y_col] = {}
		for time in ['8wks','both','baseline']:
			model_dict_slopeint_top10[name][y_col][time]={}
			y=df_clean[y_col]
			X=df_clean[top15features[name][y_col][time]]
			non_numeric_cols = X.select_dtypes(include=['object']).columns
			if len(non_numeric_cols) > 0:
				print("Non-numeric columns found:", non_numeric_cols)
				print(X[non_numeric_cols].head())

				
			# Inputs
			groups = df_clean['num_id']
			gkf = GroupKFold(n_splits=5)

			for model_name in models:
				model_dict_slopeint_top10[name][y_col][time][model_name] = {}
				model_dict_slopeint_top10[name][y_col][time][model_name]['model']=[]
				model_dict_slopeint_top10[name][y_col][time][model_name]['X_test']=[]
				model_dict_slopeint_top10[name][y_col][time][model_name]['r2_scores']=[]
				model_dict_slopeint_top10[name][y_col][time][model_name]['mae_scores']=[]
				for train_idx, test_idx in gkf.split(X, y, groups):
					X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
					y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

					if X_train.shape[0]==0 or X_train.shape[1]==0 or len(X_train)==0:
						continue

					model = models[model_name][0]
					if 'bin' in y_col: #binary-- use classifier model instead of regressor model
						model= models[model_name][1]
					model.fit(X_train, y_train)
					model_dict_slopeint_top10[name][y_col][time][model_name]['model'].append(model)

					y_pred = np.round(model.predict(X_test)).astype(int) #ordinal Y
					model_dict_slopeint_top10[name][y_col][time][model_name]['X_test'].append(X_test)

					r2 = r2_score(y_test, y_pred)
					model_dict_slopeint_top10[name][y_col][time][model_name]['r2_scores'].append(r2)

					mae = mean_absolute_error(y_test, y_pred)
					model_dict_slopeint_top10[name][y_col][time][model_name]['mae_scores'].append(mae)
					

				print(f"TIME {time} -- Num features: {len(X_train.columns)}, train: {len(X_train)} obs, test: {len(X_test)} obs || mean R²: {round(np.mean(model_dict_slopeint_top10[name][y_col][time][model_name]['r2_scores']), 4)}, mean MAE: {round(np.mean(model_dict_slopeint_top10[name][y_col][time][model_name]['mae_scores']), 4)} of {round(y.min(), 1)}-{round(y.max(), 1)}\n")




