print('testing')
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, make_scorer, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import randint, uniform
from sklearn.model_selection import GroupKFold, cross_val_score


from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier
from catboost import Pool
import catboost as cb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from itertools import product,chain

import plotly.express as px

from scipy.stats import pearsonr

print('importing shap')
import shap
print('shap imported, importing initjs')
shap.initjs()
print('initjs imported')
# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add project root to sys.path
sys.path.append(project_root)
# Define data directory
brighten_dir = os.path.join(project_root, 'BRIGHTEN_data')
charts_dir = os.path.join(project_root, 'charts')
shap_dir = os.path.join(charts_dir, 'shap')
cv_dir = os.path.join(charts_dir, 'cv_scores')
val_dir = os.path.join(charts_dir, 'val_scores')

# Import and reload my custom scripts
from scripts import preprocessing as pre
from scripts import visualization as vis
from scripts import feature_selection as fs
from scripts import clustering as cl
from scripts import variables as var
importlib.reload(pre)
importlib.reload(vis)
importlib.reload(fs)
importlib.reload(cl)
importlib.reload(var)

print('custom scrips imported')
# # Import from cloned github repos
# import hyperopt
# print(hyperopt.__file__)
# from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
# from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
# import hyperopt.pyll.stochastic
################ DEFINE column variables from data ###################
from scripts.variables import id_columns, daily_cols_v1, daily_v2_common 
from scripts.variables import phq2_cols, phq9_cols, weekly_cols, passive_cols, survey_cols

## Load in dfs scaled
df_names = ['v1_day', 'v2_day', 'v1_week', 'v2_week']
df_pca = ['v1_day_pca', 'v1_week_pca']
df_all = df_names + df_pca

if not 'idx' in id_columns:
    id_columns.append('idx')
print(id_columns)


########################################## MODELS #######################################
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import LinearSVC, LinearSVR
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb
# commenting out linear svm because we didn't drop skewed/two tailed variables
regs = [
    ('Decision Tree', DecisionTreeRegressor()),
    #('Linear SVM', LinearSVR(random_state=42, max_iter=10000, dual='auto')),
    ('Hist Gradient Tree', HistGradientBoostingRegressor()),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('XGBoost', XGBRegressor(objective='reg:squarederror', random_state=42)),
    ('AdaBoost', AdaBoostRegressor(random_state=42)),
    ('Gradient Boost', GradientBoostingRegressor(random_state=42)),
    ('Bagging', BaggingRegressor(random_state=42)),
    ('CatBoost', CatBoostRegressor(random_state=42, verbose=0))
]


regs_someNA = [
    ('Hist Gradient Boost', HistGradientBoostingRegressor()),
    ('XGBoost', XGBRegressor(objective='reg:squarederror', random_state=42)),
    ('CatBoost', CatBoostRegressor(random_state=42, verbose=0))
]



print('testing')

cv_results = {}
test_results = {}

for name in df_names:
    print(name)
    cv_results[name] = {}
    test_results[name] = {}
    X_train = pd.read_csv(os.path.join(brighten_dir, f'{name}_X_train.csv'))
    y_train= pd.read_csv(os.path.join(brighten_dir, f'{name}_y_train.csv'))

    # Create KFold object which groups based on num ID
    gkf = GroupKFold(n_splits=5)
    groups = X_train['num_id']    # Set num_id for the groups

    # Now drop all other id columns 
    X_train_cleaned = X_train.drop(columns=[col for col in X_train.columns if col in id_columns or 'Unnamed' in col or '_int' in col])
    y_train_cleaned = y_train.drop(columns=[col for col in y_train.columns if 'Unnamed' in col]).squeeze() #squeeze into a list from a series
    print(f'Training shapes: y {y_train_cleaned.shape}, X {X_train_cleaned.shape}')
    # Evaluate each classifier using cross-validation

    for reg_name, reg in regs:
        print(f'{name}, {reg_name}')
        cv_results[name][reg_name] = {}
        test_results[name][reg_name] = {}

        cv_results[name][reg_name]['r2'] = []
        cv_results[name][reg_name]['mae'] = []
        cv_results[name][reg_name]['pmc'] = []

        test_results[name][reg_name]['r2'] = []
        test_results[name][reg_name]['mae'] = []
        test_results[name][reg_name]['pmc'] = []

        # 5 fold cross validation
        gkf = GroupKFold(n_splits=5)
        # going through each CV split
        for train_idx, val_idx in gkf.split(X_train_cleaned, y_train_cleaned, groups=groups):
            X_tr, X_cv_val = X_train_cleaned.iloc[train_idx], X_train_cleaned.iloc[val_idx]
            y_tr, y_cv_val = y_train_cleaned.iloc[train_idx], y_train_cleaned.iloc[val_idx]
            
            reg.fit(X_tr, y_tr)  # Now this reg is the trained model for this fold
            # Predict on CV
            y_cv_val_pred = reg.predict(X_cv_val)
            # Evaluate r2 CV score
            cv_score = r2_score(y_cv_val, y_cv_val_pred)
            cv_mae = mean_absolute_error(y_cv_val, y_cv_val_pred)
            cv_pmc = pearsonr(y_cv_val, y_cv_val_pred)
            print(f'cv score: {cv_score}, mae {cv_mae} PMC: {cv_pmc[0]:.3f}, p-val {cv_pmc[1]:.3f}')
            cv_results[name][reg_name]['r2'].append(cv_score)
            cv_results[name][reg_name]['mae'].append(cv_mae)
            cv_results[name][reg_name]['pmc'].append(cv_pmc[0])

            


        
        # For that reg, predict test
        X_val = pd.read_csv(os.path.join(brighten_dir, f'{name}_X_val.csv'))
        X_val_cleaned = X_val.drop(columns=[col for col in X_val.columns if col in id_columns or 'Unnamed' in col or '_int' in col])
        y_val = pd.read_csv(os.path.join(brighten_dir, f'{name}_y_val.csv'))
        y_val_cleaned = y_val.drop(columns=[col for col in y_val.columns if 'Unnamed' in col]).squeeze() #squeeze into a list from a series
        # Fit that reg
        reg.fit(X_train_cleaned, y_train_cleaned)
        y_val_pred = reg.predict(X_val_cleaned)
        # Evaluate r2 test score
        test_score = r2_score(y_val_cleaned, y_val_pred)
        test_mae = mean_absolute_error(y_val_cleaned, y_val_pred)
        test_pmc = pearsonr(y_val_cleaned, y_val_pred)
        print(f'R² on test for {reg_name}: {test_score:.3f}, PMC: {test_pmc[0]:.3f}, p-val {test_pmc[1]:.3f}')
        test_results[name][reg_name]['r2'].append(test_score)
        test_results[name][reg_name]['mae'].append(test_mae)
        test_results[name][reg_name]['pmc'].append(test_pmc[0])
        if 'Ada' not in reg_name and "agging" not in reg_name:
            explainer = shap.Explainer(reg)
            shap_values = explainer.shap_values(X_val_cleaned)
            shap.summary_plot(shap_values, X_val_cleaned,show=False)
            plt.savefig(os.path.join(shap_dir, f"shap_{name}_{reg_name}_imputed.png"), bbox_inches="tight", dpi=300)
            plt.clf()   # (Optional) Clear the figure to avoid overlap with future plots




# Extract test data to flattened df for plotting
flat_cv_results = []

# iterate over each dataset name
for dataset_name, reg_results in cv_results.items():
    # iterate over the scores within it
    for reg_name, metrics in reg_results.items():
            for metric in metrics.keys():
                for cv_score in metrics[metric]:
                    flat_cv_results.append({
                        'Dataset': dataset_name,
                        'Regressor': reg_name,
                        'Metric': metric,
                        'Score': cv_score,
                    })
flat_cv_results_df = pd.DataFrame(flat_cv_results)
flat_cv_results_df.to_csv(os.path.join(brighten_dir, 'flat_cv_results_df.csv'))
plt.figure(figsize=(14, 8))
sns.boxplot(x='Regressor', y='Score', hue='Dataset', data=flat_cv_results_df[(flat_cv_results_df['Metric']=='pmc') & (flat_cv_results_df['Regressor']!='Decision Tree')])
plt.xticks(rotation=45)
plt.title('PMC/Pearson R CV Scores by Regressor and Dataset, imputation')
plt.savefig(os.path.join(cv_dir, 'PMC/Pearson R CV Scores by Regressor and Dataset on _train dataset, imputed, before tuning.png'))


test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv(os.path.join(brighten_dir, 'test_results_df.csv'))




# Extract test data to flattened df for plotting
flat_test_results = []

# iterate over each dataset name
for dataset_name, reg_results in test_results.items():
    # iterate over the scores within it
    for reg_name, metrics in reg_results.items():
            for metric in metrics.keys():
                flat_test_results.append({
                    'Dataset': dataset_name,
                    'Regressor': reg_name,
                    'Metric': metric,
                    'Score': metrics[metric][0],
                })
flat_test_results_df = pd.DataFrame(flat_test_results)


plt.figure(figsize=(14, 8))
sns.barplot(x='Regressor', y='Score', hue='Dataset', data=flat_test_results_df[flat_test_results_df['Metric']=='pmc'])
plt.xticks(rotation=45)
plt.title('PMC/Pearson R Test Scores by Regressor and Dataset, imputation')
plt.savefig(os.path.join(val_dir, 'PMC/Pearson R Test Scores by Regressor and Dataset on _val dataset, imputed, before tuning.png'))
