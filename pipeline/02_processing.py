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
import datetime as dt
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
#from xgboost import XGBClassifier, XGBRegressor
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

importlib.reload(paths)
importlib.reload(pre)
importlib.reload(vis)
importlib.reload(variables)
importlib.reload(fs)

# Filepaths
brighten_dir = paths.DATA

################ DEFINE column variables from data ###################
from scripts.variables import id_columns
from scripts.variables import all_cols, all_daily_cols, weekly_cols, baseline_cols, drop_weekly_cols
from scripts.variables import daily_cols_v1, daily_v2_sensor_hr, daily_v2_weather, daily_cols_v2 
from scripts.variables import gad_cols, phq9_base, alc_cols, phq9_cols, phq2_cols, sleep_cols, gic_cols, sds_cols


# Define label variables
df_names = ['v1_day', 'v2_day', 'v1_week', 'v2_week']
aggregate_dfs = ['alldays_df','week_df']
# Update endings list if order changes


import warnings
warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values",
    category=UserWarning,
    module="sklearn.impute._base"
)


# Defining the transformations
yj_pipeline = Pipeline(steps=[
    ('power', PowerTransformer(method='yeo-johnson')),
    ('scale', StandardScaler())
])

bc_pipeline = Pipeline(steps=[
    ('power', PowerTransformer(method='box-cox')),
    ('scale', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

ordinal_pipeline = Pipeline(steps=[
    ('scale', StandardScaler())
])

non_skewed_pipeline = Pipeline(steps=[
    ('scale', StandardScaler())
])


########################################## MODELS #######################################


# Models
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
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
    # 'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
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

# %% [markdown]
# my pipeline: 
# 1. remove outliers
# 4. apply log transform to skewed or wide-tailed data
# 5. remove too-skewed data
# 6. Impute
# 7. Scale
# 8. Drop columns with too few unique values or too little variance

# %%
## Investigate kurtosis and skewedness
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

skewed_cols = {}
for name in df_names:
    skewed_cols[name] = {}
    df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval.csv'))
    print(f'\n\nFor {name}:')
    numeric_cols = df.select_dtypes(include=('int64','float64')).columns.to_list()
    non_bin_cols = [col for col in numeric_cols if '_bin' not in col and "_indicator" not in col and "_missing" not in col and "nonzero" not in col]
    
    skew_list = df[non_bin_cols].skew(numeric_only=True).sort_values(ascending=False) # sort by highest, display    
    if len(skew_list[skew_list > 1])>0:
        skewed_cols[name]['skew'] = skew_list[skew_list > 1].index
        print(f'Of {len(skew_list)} measures, {len(skew_list[skew_list > 1])} measures have skew > 1:')
        display(skew_list[skew_list > 1])

    # Calculate kurtosis for numeric columns
    kurtosis_vals = df[non_bin_cols].kurtosis(numeric_only=True)
    kurtosis_sorted = kurtosis_vals.sort_values(ascending=False) # Sort by highest kurtosis
    skewed_cols[name]['kurtosis'] = kurtosis_sorted[kurtosis_sorted > 2].index
    
    if len(kurtosis_sorted[kurtosis_sorted > 2])>0:
        print(f'Of {len(kurtosis_sorted)} measures, {len(kurtosis_sorted[kurtosis_sorted > 2])} measures have Kurtosis > 2:')
        display(kurtosis_sorted[kurtosis_sorted > 2]) #display

    if len(kurtosis_sorted[kurtosis_sorted.isna()])>0:
        print(f'{len(kurtosis_sorted[kurtosis_sorted.isna()])} columns have NaN in Kurtosis:')
        display(kurtosis_sorted[kurtosis_sorted.isna()]) #display

    # More investigation into kurtosis NaN values
    for col in kurtosis_sorted[kurtosis_sorted.isna()].index:
        print(f'Kurtosis is NaN for {col}:')
        if col in df.columns:
            print('Unique values:', df[col].nunique())      # Unique values
            print("Missing values:", df[col].isna().sum())   # Missing values
            print("Variance:", df[col].var())     # Summary stats



# %% [markdown]
# # Do preprocessing column transformer on long data before making it wide

# %%
id_columns

# %%
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

target_columns = phq2_cols + phq9_cols
to_categorical = ['race','gender','marital_status', 'season', 'cohort']
ordinal_columns = list(set(gad_cols + phq9_base + alc_cols + sleep_cols + gic_cols + sds_cols +  ['education', 'income_satisfaction','incomelastyear'])) 

box_cox_columns = ['mobility','mobility_radius']
yeo_johnson_columns = list(set([col for col in skewed_cols[name]['skew'].to_list()+skewed_cols[name]['kurtosis'].to_list() if col not in box_cox_columns+target_columns+to_categorical+id_columns+ordinal_columns+baseline_cols and 'dt' not in col]))
non_skewed_columns = list(set([col for col in all_daily_cols if col not in box_cox_columns+target_columns+to_categorical+yeo_johnson_columns+ordinal_columns+id_columns and 'dt' not in col]))

def is_yj_safe(series):
    """Check if a column is safe to apply Yeo-Johnson to."""
    s = series.dropna()
    if len(s) < 3:
        return False
    if s.nunique() < 3:  # nearly constant
        return False
    if s.std() == 0:  # zero variance
        return False
    return True



for split in ['trainval','test']:
    for name in df_names:
        print(f"\n=== Processing: {name} ===")
        df = pd.read_csv(os.path.join(brighten_dir, f'{name}_{split}.csv'), low_memory=False)
        df = df.dropna(axis=1, how='all') # drop any columns which are fully NaN
        df['date'] = pd.to_datetime(df['date'])
        print('id columns originally:', [col for col in df.columns if col in id_columns])

        # Columns selected to each transformer
        cat_cols = [item for item in df.columns 
                    if any(term in item for term in to_categorical) and 'dt' not in item]
        yj_cols = [
            item for item in df.columns 
            if any(term in item for term in yeo_johnson_columns) and 'dt' not in item
            and is_yj_safe(df[item])  # add this check
        ]
        bc_cols = [item for item in df.columns 
                   if any(term in item for term in box_cox_columns) and 'dt' not in item]
        non_skewed_cols = [item for item in df.columns 
                           if any(term in item for term in non_skewed_columns) and 'dt' not in item]
        ordinal_cols = [col for col in df.columns 
                        if any(term in col for term in ordinal_columns) and 'dt' not in col]
        target_cols = [col for col in df.columns 
                       if any(term in col for term in target_columns) and 'base' not in col and 'dt' not in col]


        for tname, cols in [('yj', yj_cols), ('bc', bc_cols), ('cat', cat_cols), ('non-skew', non_skewed_cols),('ordinal', ordinal_cols), ('target', target_cols)]:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                print(f"Columns listed in {tname} but missing from df: {missing}")

            for col in cols:
                if col in df.columns:
                    print(f'{tname} | {col} | dtype: {df[col].dtype} | sample: {df[col].dropna().iloc[0]}')



# %%
for name in df_names:
    fitted_processor = None  # store fitted preprocessor here
    for split in ['trainval','test']:
        
        print(f"\n=== Processing: {name} ===")
        df = pd.read_csv(os.path.join(brighten_dir, f'{name}_{split}.csv'), low_memory=False)
        df['date'] = pd.to_datetime(df['date'])
        print('id columns originally:', [col for col in df.columns if col in id_columns])

        # Columns selected to each transformer
        cat_cols = [item for item in df.columns 
                    if any(term in item for term in to_categorical) and 'dt' not in item]
        yj_cols = [
            item for item in df.columns 
            if any(term in item for term in yeo_johnson_columns) and 'dt' not in item
            and is_yj_safe(df[item])  # add this check
        ]
        bc_cols = [item for item in df.columns 
                   if any(term in item for term in box_cox_columns) and 'dt' not in item]
        non_skewed_cols = [item for item in df.columns 
                           if any(term in item for term in non_skewed_columns) and 'dt' not in item]
        ordinal_cols = [col for col in df.columns 
                        if any(term in col for term in ordinal_columns) and 'dt' not in col]
        target_cols = [col for col in df.columns 
                       if any(term in col for term in target_columns) and 'base' not in col and 'dt' not in col]
                
        transformers = []

        if len(yj_cols) > 0:
            transformers.append(('yj', yj_pipeline, yj_cols))
        if len(bc_cols) > 0:
            transformers.append(('bc', bc_pipeline, bc_cols))
        if len(cat_cols) > 0:
            transformers.append(('cat', categorical_pipeline, cat_cols))
        if len(non_skewed_cols) > 0:
            transformers.append(('non-skew', non_skewed_pipeline, non_skewed_cols))

        print(f'There are {len(transformers)} of 5 transformer pipelines')

        if split == 'trainval':
            # Fit only on training data
            fitted_preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            ).set_output(transform='pandas')
            fitted_preprocessor.fit(df)

        transformed_df = fitted_preprocessor.transform(df)

        transformed_df.columns = transformed_df.columns.str.replace('non-skew__', '', regex=False)
        transformed_df.columns = transformed_df.columns.str.replace('yj__', '', regex=False)
        transformed_df.columns = transformed_df.columns.str.replace('bc__', '', regex=False)
        transformed_df.columns = transformed_df.columns.str.replace('cat__', '', regex=False)
        transformed_df.columns = transformed_df.columns.str.replace('remainder__', '', regex=False)
        transformed_df = transformed_df.loc[:, ~transformed_df.columns.str.contains('_nan')]
        transformed_df = transformed_df.loc[:, ~transformed_df.columns.str.contains('Unnamed')]

        # Reorder id columns
        id_columns_intact = [col for col in id_columns if col in transformed_df.columns]
        transformed_df = transformed_df[id_columns_intact + [col for col in transformed_df.columns if col not in id_columns_intact]]

        # There's no one from this race in test_df and it's giving me problems with my predictive models later, and it's a column addded by cat encoder, so i just added it and said no one is this race by making it all = 0
        if 'test' in split:
            transformed_df['race_5.0'] = 0
            transformed_df['race_1.0'] = 0
            transformed_df['race_7.0'] = 0		
        

        transformed_df.to_csv(os.path.join(brighten_dir, f'{name}_{split}_nonskew.csv'))
        display(transformed_df)
        print(f'Saved {name}_{split}_nonskew.csv to brighten_dir')


print('Last run:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))


# %%
## Investigate kurtosis and skewedness
skewed_cols = {}
for name in df_names:
    skewed_cols[name] = {}
    df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_nonskew.csv'))
    print(f'\n\nFor {name}:')
    numeric_cols = [col for col in df.columns.to_list() if col in all_daily_cols+baseline_cols+weekly_cols+phq9_cols+phq2_cols]
    non_bin_cols = [col for col in numeric_cols if '_bin' not in col and "_indicator" not in col and "_missing" not in col and "nonzero" not in col]
    
    skew_list = df[non_bin_cols].skew(numeric_only=True).sort_values(ascending=False) # sort by highest, display    
    if len(skew_list[skew_list > 1])>0:
        skewed_cols[name]['skew'] = skew_list[skew_list > 1]
        print(f'Of {len(skew_list)} measures, {len(skew_list[skew_list > 1])} measures have skew > 1:')
        print(skew_list[skew_list > 1])

    # Calculate kurtosis for numeric columns
    kurtosis_vals = df[non_bin_cols].kurtosis(numeric_only=True)
    kurtosis_sorted = kurtosis_vals.sort_values(ascending=False) # Sort by highest kurtosis
    skewed_cols[name]['kurtosis'] = kurtosis_sorted[kurtosis_sorted > 2]
    if len(kurtosis_sorted[kurtosis_sorted > 2])>0:
        print(f'Of {len(kurtosis_sorted)} measures, {len(kurtosis_sorted[kurtosis_sorted > 2])} measures have Kurtosis > 2:')
        print(kurtosis_sorted[kurtosis_sorted > 2]) #display

    if len(kurtosis_sorted[kurtosis_sorted.isna()])>0:
        print(f'{len(kurtosis_sorted[kurtosis_sorted.isna()])} columns have NaN in Kurtosis:')
        print(kurtosis_sorted[kurtosis_sorted.isna()]) #display

    # More investigation into kurtosis NaN values
    for col in kurtosis_sorted[kurtosis_sorted.isna()].index:
        print(f'Kurtosis is NaN for {col}:')
        if col in df.columns:
            print('Unique values:', df[col].nunique())      # Unique values
            print("Missing values:", df[col].isna().sum())   # Missing values
            print("Variance:", df[col].var())     # Summary stats

print('Last run:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))


# %% [markdown]
# # Standard Scale all numeric non-categorical columns

# %%
from sklearn.compose import ColumnTransformer
scaler = StandardScaler()
cols = 	yeo_johnson_columns+box_cox_columns+non_skewed_columns+ordinal_columns+target_columns


for name in df_names:
    fitted_processor = None  # store fitted preprocessor here
    for split in ['trainval','test']:
        skewed_cols[name] = {}
        df = pd.read_csv(os.path.join(brighten_dir, f'{name}_{split}_nonskew.csv'))
        cols_present = [col for col in cols if col in df.columns]
        df_cols_present = df[cols_present]
        df_cols_numeric = df[cols_present].select_dtypes(include=('int64','float64'))
        cols_not_numeric = [col for col in df_cols_present if col not in df_cols_numeric]
        if len(cols_not_numeric) > 0:
            print(f'Warning: these cols were not numeric: {cols_not_numeric}, skipping')
        
        if split=='trainval':
            fitted_processor = ColumnTransformer([('standard', scaler, df_cols_numeric.columns.to_list())], remainder = 'passthrough').set_output(transform='pandas')
            fitted_processor.fit(df)

        scaled_df = fitted_processor.transform(df)
        print(f'Scaled cols: {[col.replace('standard__','') for col in scaled_df.columns if 'standard' in col]}')
        scaled_df.columns = scaled_df.columns.str.replace('standard__', '')
        scaled_df.columns = scaled_df.columns.str.replace('remainder__', '')
        display(scaled_df)
        scaled_df.to_csv(os.path.join(brighten_dir, f'{name}_{split}_transformed.csv'), index=False)

print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))


# %% [markdown]
# # Make imputed version of the dataset
# People are already filtered to have >70% of data
# Imputing their mean so that we can use PCA and other non-NA methods

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

numeric_cols = 	yeo_johnson_columns+box_cox_columns+non_skewed_columns+ordinal_columns+target_columns

imputed_df = pd.DataFrame()
for name in df_names:
    count=0
    for split in ['trainval','test']:
        df = pd.read_csv(os.path.join(brighten_dir, f'{name}_{split}_transformed.csv'))
        numeric_cols_present = [col for col in numeric_cols if col in df.columns]
        df_cols_present = df[numeric_cols_present]
        df_cols_numeric = df[numeric_cols_present].select_dtypes(include=('int64','float64'))
        cols_not_numeric = [col for col in df_cols_present if col not in df_cols_numeric]
        if len(cols_not_numeric) > 0:
            print(f'Warning: these cols were not numeric: {cols_not_numeric}, skipping')

        # Skip imputing phq9 cols so we don't impute our target...
        cols_to_impute = [col for col in df_cols_numeric.columns if col not in phq9_cols]
        imputer_pipe = ColumnTransformer([('impute', imputer, cols_to_impute)], remainder = 'passthrough').set_output(transform='pandas')

        all_imputed = []  # collect per-subject results

        for count, (sub, sub_df) in enumerate(df.groupby('num_id')):
            imputed_sub_df = imputer_pipe.fit_transform(sub_df)

            # clean up column names once per loop
            imputed_sub_df.columns = imputed_sub_df.columns.str.replace('impute__', '', regex=False)
            imputed_sub_df.columns = imputed_sub_df.columns.str.replace('remainder__', '', regex=False)

            imputed_sub_df['num_id'] = sub
            all_imputed.append(imputed_sub_df)

            if count == 0:
                imputed_cols = [col for col in imputed_sub_df.columns if col in cols_to_impute]
                print(f'Imputed cols for {name}: {imputed_cols}')
                count+=1

        # concatenate all subjects for this file
        imputed_df = pd.concat(all_imputed, ignore_index=True)

        display(imputed_df[[col for col in id_columns if col in imputed_df.columns]+[col for col in imputed_df.columns if col not in id_columns]].head())

        # save output
        out_path = os.path.join(brighten_dir, f'{name}_{split}_imputed.csv')
        imputed_df.to_csv(out_path, index=False)
        print(f'Saved imputed data to {out_path}')



# %% [markdown]
# # WIDE DF - takes a while to run

# %%
# Wide df: Define coverage and only keep subjects where they have >=threshold% of daily data over 12 weeks
coverage_threshold=0.5
filter_for_coverage = False
results=[]
for v in ['v1', 'v2']:
    for time in ['day', 'week']:
        print(f"\n=== Processing: {v} {time} ===")
        fname = f'{v}_{time}_trainval_transformed.csv'
        df = pd.read_csv(os.path.join(brighten_dir, fname), low_memory=False)

        df_filtered_byDay = df.loc[df['day']<56] # only keep 8weeks of data

        ##### Create wide df of daily cols ######
        sensor_columns = daily_cols_v1 if v == 'v1' else daily_v2_sensor_hr + daily_v2_weather
        sensor_columns_present = [col for col in df_filtered_byDay.columns if any(term in col for term in sensor_columns)]

        dailySurvey_cols = phq2_cols + sensor_columns
        daily_cols_present = [col for col in df_filtered_byDay.columns if any(term in col for term in dailySurvey_cols)]
        #print(daily_cols_present)

        indicator_cols = [f'{col}_indicator' for col in sensor_columns]
        indicator_cols_present = [col for col in df_filtered_byDay.columns if any(term in col for term in indicator_cols)]

        # Only keep subjects where they have >=70% of daily data over 12 weeks
        if filter_for_coverage:
            grouped = df_filtered_byDay.groupby('num_id')[sensor_columns_present].apply(lambda x: x.notna().mean(axis=1).mean())
            keep_ids_days = grouped[grouped >= coverage_threshold].index
            df_filtered_bySub = df_filtered_byDay[df_filtered_byDay['num_id'].isin(keep_ids_days)]
            print(f'{df_filtered_byDay['num_id'].nunique()} subs before >{coverage_threshold} daily filtering, {df_filtered_bySub['num_id'].nunique()} subs after >{coverage_threshold} daily filtering,')
        else:
            df_filtered_bySub = df_filtered_byDay

        daily_df_wide = pd.DataFrame()
        for id in df_filtered_bySub['num_id'].unique():
            df_sub = df_filtered_bySub[df_filtered_bySub['num_id'] == id].copy()
            df_sub['day'] = df_sub['day'].astype(int)
            df_wide = df_sub.set_index(['day','num_id'])[daily_cols_present].unstack(level='day')
            df_wide.columns = [f"{col}_{day}" for col, day in df_wide.columns]
            df_wide = df_wide.reset_index()
            if df_wide.shape[0] > 1:
                print('df_wide')
                display(df_wide)
                df_wide_filled = df_wide.bfill().ffill().drop_duplicates()
                print('df_wide_filled')
                display(df_wide_filled)
            else:
                df_wide_filled = df_wide

            daily_df_wide = pd.concat([daily_df_wide, df_wide])

        
        assert daily_df_wide['num_id'].is_unique, "Duplicate num_id rows in daily_df_wide"


        ##### Create wide df of weekly cols ######
        weekly_cols_present = [col for col in weekly_cols if col in df.columns]


        weekly_df_wide = pd.DataFrame()
        for id in df_filtered_byDay['num_id'].unique():
            df_sub = df_filtered_byDay[df_filtered_byDay['num_id'] == id].copy()
            df_sub['week'] = df_sub['week'].astype(int)
            df_agg = df_sub.groupby(['num_id', 'week'])[weekly_cols_present].mean()
            df_wide = df_agg.unstack(level='week')
            df_wide.columns = [f"{col}_week{week}" for col, week in df_wide.columns]
            df_wide = df_wide.reset_index()
            if df_wide.shape[0] > 1:
                df_wide_filled = df_wide.bfill().ffill().drop_duplicates()
            else:
                df_wide_filled=df_wide
            #present_drop_weekly_cols = [col for col in drop_weekly_cols if col in df_wide_filled.columns.to_list()]
            #df_wide_filled = df_wide_filled.drop(columns=present_drop_weekly_cols)
            #print('for sub', id, 'coverage:', df_wide_filled.columns.notna().sum()/df_wide_filled.shape[1])
            if filter_for_coverage:
                if df_wide_filled.notna().mean(axis=1)[0] > coverage_threshold:
                    weekly_df_wide = pd.concat([weekly_df_wide, df_wide_filled])
            else:
                weekly_df_wide = pd.concat([weekly_df_wide, df_wide_filled])

        if filter_for_coverage:
            print(f'{df_filtered_byDay['num_id'].nunique()} subs before >{coverage_threshold} weekly filtering, {weekly_df_wide['num_id'].nunique()} subs after >{coverage_threshold} weekly filtering,')
        
        assert weekly_df_wide['num_id'].is_unique, "Duplicate num_id rows in weekly_df_wide"

        ##### Create wide df of baseline cols ######
        baseline_cols_present = [col for col in df.columns if any(item in col for item in baseline_cols+['date','num_id'])]
        df_base = df[baseline_cols_present]
        print(baseline_cols_present)
        df_subs_full=pd.DataFrame()
        df_subs = df_base.groupby('num_id')
        for sub, sub_df in df_subs:
            sub_df_filled = sub_df.bfill().ffill().infer_objects(copy=False).drop(columns='date').drop_duplicates()
            if sub_df_filled.shape[0] > 1:
                for col in sub_df_filled.columns:
                    if len(sub_df_filled[col].unique()) > 1:
                        sub_df_filled[col] = sub_df_filled[col].unique()[0]
                sub_df_filled = sub_df_filled.drop_duplicates()
            if sub_df_filled.shape[0] > 1:
                raise ValueError("Multiple rows found for a subject in baseline that should be 1-row")
            
            df_subs_full = pd.concat([df_subs_full, sub_df_filled])
        # print('df_subs_full')
        # display(df_subs_full[['num_id']+[col for col in df_subs_full if col != 'num_id']])

        if filter_for_coverage:
            print(f'{df_subs_full['num_id'].nunique()} subs in df_subs_full, {weekly_df_wide['num_id'].nunique()} subs in weekly_df_wide, {daily_df_wide['num_id'].nunique()} subs in daily_df_wide')
        
        assert df_subs_full['num_id'].is_unique, "Duplicate num_id rows in df_subs_full"


        ##### Combine base, weekly and daily ######
        df_combined = weekly_df_wide.merge(df_subs_full, on='num_id', how='left')
        df_combined = df_combined.merge(daily_df_wide, on='num_id', how='left')
        df_combined = df_combined.loc[:, ~df_combined.columns.astype(str).str.contains('^Unnamed')]
        print('df_combined shape:', df_combined.shape, 'df_combined subs:', df_combined['num_id'].nunique())
        assert df_combined['num_id'].is_unique, "Duplicate num_id rows in df_combined"
        

        df_combined.to_csv(os.path.join(brighten_dir, f'{v}_{time}_Xy_8wks_wide_trainval.csv'), index=False)


# %%
# # ?is this a duplicate?? Wide df: Define coverage and only keep subjects where they have >=threshold% of daily data over 12 weeks
coverage_threshold=0.5
filter_for_coverage = False
results=[]
for v in ['v1', 'v2']:
    for time in ['day', 'week']:
        print(f"\n=== Processing: {v} {time} ===")
        fname = f'{v}_{time}_trainval_transformed.csv'
        df = pd.read_csv(os.path.join(brighten_dir, fname), low_memory=False)

        df_filtered_byDay = df.loc[df['day']<56] # only keep 8weeks of data

        ##### Create wide df of daily cols ######
        sensor_columns = daily_cols_v1 if v == 'v1' else daily_v2_sensor_hr + daily_v2_weather
        sensor_columns_present = [col for col in df_filtered_byDay.columns if any(term in col for term in sensor_columns)]

        dailySurvey_cols = phq2_cols + sensor_columns
        daily_cols_present = [col for col in df_filtered_byDay.columns if any(term in col for term in dailySurvey_cols)]
        #print(daily_cols_present)

        indicator_cols = [f'{col}_indicator' for col in sensor_columns]
        indicator_cols_present = [col for col in df_filtered_byDay.columns if any(term in col for term in indicator_cols)]

        # Only keep subjects where they have >=70% of daily data over 12 weeks
        if filter_for_coverage:
            grouped = df_filtered_byDay.groupby('num_id')[sensor_columns_present].apply(lambda x: x.notna().mean(axis=1).mean())
            keep_ids_days = grouped[grouped >= coverage_threshold].index
            df_filtered_bySub = df_filtered_byDay[df_filtered_byDay['num_id'].isin(keep_ids_days)]
            print(f'{df_filtered_byDay['num_id'].nunique()} subs before >{coverage_threshold} daily filtering, {df_filtered_bySub['num_id'].nunique()} subs after >{coverage_threshold} daily filtering,')
        else:
            df_filtered_bySub = df_filtered_byDay

        daily_df_wide = pd.DataFrame()
        for id in df_filtered_bySub['num_id'].unique():
            df_sub = df_filtered_bySub[df_filtered_bySub['num_id'] == id].copy()
            df_sub['day'] = df_sub['day'].astype(int)
            df_wide = df_sub.set_index(['day','num_id'])[daily_cols_present].unstack(level='day')
            df_wide.columns = [f"{col}_{day}" for col, day in df_wide.columns]
            df_wide = df_wide.reset_index()
            if df_wide.shape[0] > 1:
                print('df_wide')
                display(df_wide)
                df_wide_filled = df_wide.bfill().ffill().drop_duplicates()
                print('df_wide_filled')
                display(df_wide_filled)
            else:
                df_wide_filled = df_wide

            daily_df_wide = pd.concat([daily_df_wide, df_wide])

        
        assert daily_df_wide['num_id'].is_unique, "Duplicate num_id rows in daily_df_wide"


        ##### Create wide df of weekly cols ######
        weekly_cols_present = [col for col in weekly_cols if col in df.columns]


        weekly_df_wide = pd.DataFrame()
        for id in df_filtered_byDay['num_id'].unique():
            df_sub = df_filtered_byDay[df_filtered_byDay['num_id'] == id].copy()
            df_sub['week'] = df_sub['week'].astype(int)
            df_agg = df_sub.groupby(['num_id', 'week'])[weekly_cols_present].mean()
            df_wide = df_agg.unstack(level='week')
            df_wide.columns = [f"{col}_week{week}" for col, week in df_wide.columns]
            df_wide = df_wide.reset_index()
            if df_wide.shape[0] > 1:
                df_wide_filled = df_wide.bfill().ffill().drop_duplicates()
            else:
                df_wide_filled=df_wide
            #present_drop_weekly_cols = [col for col in drop_weekly_cols if col in df_wide_filled.columns.to_list()]
            #df_wide_filled = df_wide_filled.drop(columns=present_drop_weekly_cols)
            #print('for sub', id, 'coverage:', df_wide_filled.columns.notna().sum()/df_wide_filled.shape[1])
            if filter_for_coverage:
                if df_wide_filled.notna().mean(axis=1)[0] > coverage_threshold:
                    weekly_df_wide = pd.concat([weekly_df_wide, df_wide_filled])
            else:
                weekly_df_wide = pd.concat([weekly_df_wide, df_wide_filled])

        if filter_for_coverage:
            print(f'{df_filtered_byDay['num_id'].nunique()} subs before >{coverage_threshold} weekly filtering, {weekly_df_wide['num_id'].nunique()} subs after >{coverage_threshold} weekly filtering,')
        
        assert weekly_df_wide['num_id'].is_unique, "Duplicate num_id rows in weekly_df_wide"

        ##### Create wide df of baseline cols ######
        baseline_cols_present = [col for col in df.columns if any(item in col for item in baseline_cols+['date','num_id'])]
        df_base = df[baseline_cols_present]
        print(baseline_cols_present)
        df_subs_full=pd.DataFrame()
        df_subs = df_base.groupby('num_id')
        for sub, sub_df in df_subs:
            sub_df_filled = sub_df.bfill().ffill().infer_objects(copy=False).drop(columns='date').drop_duplicates()
            if sub_df_filled.shape[0] > 1:
                for col in sub_df_filled.columns:
                    if len(sub_df_filled[col].unique()) > 1:
                        sub_df_filled[col] = sub_df_filled[col].unique()[0]
                sub_df_filled = sub_df_filled.drop_duplicates()
            if sub_df_filled.shape[0] > 1:
                raise ValueError("Multiple rows found for a subject in baseline that should be 1-row")
            
            df_subs_full = pd.concat([df_subs_full, sub_df_filled])
        # print('df_subs_full')
        # display(df_subs_full[['num_id']+[col for col in df_subs_full if col != 'num_id']])

        if filter_for_coverage:
            print(f'{df_subs_full['num_id'].nunique()} subs in df_subs_full, {weekly_df_wide['num_id'].nunique()} subs in weekly_df_wide, {daily_df_wide['num_id'].nunique()} subs in daily_df_wide')
        
        assert df_subs_full['num_id'].is_unique, "Duplicate num_id rows in df_subs_full"


        ##### Combine base, weekly and daily ######
        df_combined = weekly_df_wide.merge(df_subs_full, on='num_id', how='left')
        df_combined = df_combined.merge(daily_df_wide, on='num_id', how='left')
        df_combined = df_combined.loc[:, ~df_combined.columns.astype(str).str.contains('^Unnamed')]
        print('df_combined shape:', df_combined.shape, 'df_combined subs:', df_combined['num_id'].nunique())
        assert df_combined['num_id'].is_unique, "Duplicate num_id rows in df_combined"
        

        df_combined.to_csv(os.path.join(brighten_dir, f'{v}_{time}_Xy_8wks_wide_trainval.csv'), index=False)


# %%

# %% [markdown]
# # Extract slope and intercept from each week of daily data
#

# %%
from scipy import stats

slope_intercept_2wks = {}
for name in df_names:
    slope_intercept_2wks[name] = []
    print(f"\n=== Processing: {name} ===")
    fname = f'{name}_trainval_imputed.csv'

    df = pd.read_csv(os.path.join(brighten_dir, fname), low_memory=False)

    baseline_cols_present = [col for col in df.columns if any(item in col for item in baseline_cols+['dt','num_id'])]

    sensor_columns = daily_cols_v1 if 'v1' in name else daily_v2_sensor_hr + daily_v2_weather
    sensor_columns_present = [item for item in df.columns if any(term in item for term in sensor_columns) and 'indicator' not in item]

    target_columns = phq9_cols + phq2_cols #change to just phq9 if you want to add phq2
    target_columns_present = [item for item in df.columns if any(term in item for term in target_columns) and 'base' not in item]

    weekly_columns_present = [item for item in df.columns if any(term in item for term in weekly_cols) and 'base' not in item]
    
    for sub, sub_df in df.groupby('num_id'):
        last_day = sub_df['day'].max()
        if pd.isna(last_day):
            print(f'Warning: {sub} has no day col. sub_df[day].max() is NaN: {sub_df['day'].to_list()}')
            continue
        num_2wk_blocks = int(last_day / 14)
        skipped={}
        for i in range(1, num_2wk_blocks+1):
            skipped[i]=[]
            block=sub_df[sub_df['day']>(i-1)*14]
            block=block[block['day']<i*14]
            # print(f'block from range {(i-1)*14} - {i*14}')
            # display(block)

            n_cols_present = len(sensor_columns_present+target_columns_present+weekly_columns_present)
            for col in sensor_columns_present+target_columns_present+weekly_columns_present:
                col_block = block[['day', col]].dropna()
                if col_block.shape[0] < 2:
                    if col_block.shape[0] == 0:
                        skipped[i].append(col)
                        slope_intercept_2wks[name].append([sub, i, col, np.nan, np.nan, np.nan, np.nan, np.nan])
                        continue
                    else:
                        mean = col_block[col].mean()
                        slope_intercept_2wks[name].append([sub, i, col, np.nan, np.nan, np.nan, mean, np.nan])
                        continue
                else:
                    x = col_block['day']
                    y = col_block[col]
                    mean_value=col_block[col].mean()
                    std = col_block[col].std()
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    slope_intercept_2wks[name].append([sub, i, col, slope, intercept, p_value, mean_value, std])


            
            if len(skipped[i])>0:
                print(f'For sub {sub}, block {i} of {num_2wk_blocks}, skipped {len(skipped[i])} cols of {n_cols_present}: {skipped[i]}')
    
    slope_intercept_2wks_df = pd.DataFrame(slope_intercept_2wks[name], columns = ['num_id', '2week_block', 'col', 'slope', 'intercept', 'p_value','mean', 'std'])
    slope_intercept_2wks_df.to_csv(os.path.join(brighten_dir, f"{name}_long_slopeintercept_2wks.csv"), index=False)
    display(slope_intercept_2wks_df.head())




print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))



            


# %%
from scipy import stats

slope_intercept_wks = {}
for name in df_names:
    for split in ['trainval','test']:
        slope_intercept_wks[name] = []
        print(f"\n=== Processing: {name} ===")
        
        df = pd.read_csv(os.path.join(brighten_dir, f'{name}_trainval_imputed.csv'), low_memory=False)

        baseline_cols_present = [col for col in df.columns if any(item in col for item in baseline_cols+['dt','num_id'])]

        sensor_columns = daily_cols_v1 if 'v1' in name else daily_v2_sensor_hr + daily_v2_weather
        sensor_columns_present = [item for item in df.columns if any(term in item for term in sensor_columns) and 'indicator' not in item]

        target_columns = phq9_cols + phq2_cols #change to just phq9 if you want to add phq2
        target_columns_present = [item for item in df.columns if any(term in item for term in target_columns) and 'base' not in item]

        weekly_columns_present = [item for item in df.columns if any(term in item for term in weekly_cols) and 'base' not in item]
        
        for sub, sub_df in df.groupby('num_id'):
            last_day = sub_df['day'].max()
            if pd.isna(last_day):
                print(f'Warning: {sub} has no day col. sub_df[day].max() is NaN: {sub_df['day'].to_list()}')
                continue
            num_wk_blocks = int(last_day / 7)
            skipped={}
            for i in range(1, num_wk_blocks+1):
                skipped[i]=[]
                block=sub_df[sub_df['day']>(i-1)*7]
                block=block[block['day']<i*7]
                # print(f'block from range {(i-1)*7} - {i*7}')
                # display(block)

                n_cols_present = len(sensor_columns_present+target_columns_present+weekly_columns_present)
                for col in sensor_columns_present+target_columns_present+weekly_columns_present:
                    col_block = block[['day', col]].dropna()
                    if col_block.shape[0] < 2:
                        if col_block.shape[0] == 0:
                            skipped[i].append(col)
                            slope_intercept_wks[name].append([sub, i, col, np.nan, np.nan, np.nan, np.nan, np.nan])
                            continue
                        else:
                            mean = col_block[col].mean()
                            slope_intercept_wks[name].append([sub, i, col, np.nan, np.nan, np.nan, mean, np.nan])
                            continue
                    else:
                        x = col_block['day']
                        y = col_block[col]
                        mean_value=col_block[col].mean()
                        std = col_block[col].std()
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        slope_intercept_wks[name].append([sub, i, col, slope, intercept, p_value, mean_value, std])


                
                if len(skipped[i])>0:
                    print(f'For sub {sub}, block {i} of {num_wk_blocks}, skipped {len(skipped[i])} cols of {n_cols_present}: {skipped[i]}')
        
        slope_intercept_wks_df = pd.DataFrame(slope_intercept_wks[name], columns = ['num_id', 'week_block', 'col', 'slope', 'intercept', 'p_value','mean', 'std'])
        slope_intercept_wks_df.to_csv(os.path.join(brighten_dir, f"{name}_long_slopeintercept_wks.csv"), index=False)
        display(slope_intercept_wks_df.head())






print(f'Run on: {dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p')}')

            


# %%
slope_intercept_2wks = pd.read_csv(os.path.join(brighten_dir, f"{name}_long_slopeintercept_2wks.csv"))
display(slope_intercept_2wks)

# %%
# Print out number of participant with sufficient blocks
for name in df_names:
    subs=0
    slope_intercept_2wks_df = pd.read_csv(os.path.join(brighten_dir, f"{name}_long_slopeintercept_2wks.csv"))
    
    for sub, sub_df in slope_intercept_2wks_df.groupby('num_id'):
        if sub_df['2week_block'].nunique() > 3:
            subs+=1
    print(name, subs)
        
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))


# %%
# Create slope and intercept wide df for 2week blocks
demo_df = pd.read_csv(os.path.join(brighten_dir, f'demographics_clean.csv'))


for name in ['v1_day', 'v2_day']:
    slope_intercept_2wks_df = pd.read_csv(
        os.path.join(brighten_dir, f"{name}_long_slopeintercept_2wks.csv")
    )

    df_wide = []

    for sub, df_sub in slope_intercept_2wks_df.groupby('num_id'):
        wide_sub = {'num_id': sub}

        for block, df_block in df_sub.groupby('2week_block'):
            for _, row in df_block.iterrows():
                col = row['col']
                wide_sub[f'{col}_block{block}_slope'] = row['slope']
                wide_sub[f'{col}_block{block}_intercept'] = row['intercept']
                wide_sub[f'{col}_block{block}_mean'] = row['mean']
                wide_sub[f'{col}_block{block}_std'] = row['std']

        df_wide.append(wide_sub)

    df_wide = pd.DataFrame(df_wide).set_index('num_id')

    y_df = pd.read_csv(os.path.join(brighten_dir, f'{name}_phq9sum_6wks.csv'))
    y_df = y_df.loc[:, ~y_df.columns.str.contains('Unnamed')]

    # Merge Features and target
    Xy = pd.merge(df_wide, y_df, on='num_id', how='left')

    Xy_demo = pd.merge(Xy, demo_df, on='num_id')

    
    Xy_demo.to_csv(os.path.join(brighten_dir, f"{name}_wide_slopeintercept_2wk_outcomes.csv"), index=False)
    display(Xy_demo)

    
print('Run on:', dt.datetime.today().strftime('%a %d %b %Y, %I:%M%p'))

