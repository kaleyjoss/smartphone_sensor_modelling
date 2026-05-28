# src/paths.py
from pathlib import Path
import os

# Whole folder. CHANGE for your system.
PROJ = "/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26"

# Project folders
DATA_FOLDER = os.path.join(PROJ, "data")
SCRIPTS = os.path.join(PROJ, "data")
NB = os.path.join(PROJ, "data")
PIPE = os.path.join(PROJ, "pipeline")
RESULTS = os.path.join(PROJ, "results","tables")
CHARTS = os.path.join(PROJ, "results","charts")
# Data folders
RAW = os.path.join(DATA_FOLDER, "raw")
DATA = os.path.join(DATA_FOLDER, "interim")
SUB_DFS = os.path.join(DATA_FOLDER, "sub_dfs")
DEMO = os.path.join(DATA_FOLDER, "demographics")

# Results folder
results_bysubject_ANOVA_dir = os.path.join(PROJ, "results","charts","model_performance","by_subject",'ANOVA15')
results_bysubject_all_dir = os.path.join(PROJ, "results","charts","model_performance","by_subject",'all_features')
results_bysubject_corr_dir = os.path.join(PROJ, "results","charts","model_performance","by_subject",'correlated')
results_bysubject_LME10_dir = os.path.join(PROJ, "results","charts","model_performance","by_subject",'LME10')
results_bysubject_sensor_dir = os.path.join(PROJ, "results","charts","model_performance","by_subject",'sensor')

os.makedirs(results_bysubject_ANOVA_dir, exist_ok=True)
os.makedirs(results_bysubject_all_dir, exist_ok=True)
os.makedirs(results_bysubject_corr_dir, exist_ok=True)
os.makedirs(results_bysubject_LME10_dir, exist_ok=True)
os.makedirs(results_bysubject_sensor_dir, exist_ok=True)


results_allsubs_ANOVA_dir = os.path.join(PROJ, "results","charts","model_performance","all_subs",'ANOVA15')
results_allsubs_all_dir = os.path.join(PROJ, "results","charts","model_performance","all_subs",'all_features')
results_allsubs_corr_dir = os.path.join(PROJ, "results","charts","model_performance","all_subs",'correlated')
results_allsubs_LME10_dir = os.path.join(PROJ, "results","charts","model_performance","all_subs",'LME10')
results_allsubs_sensor_dir = os.path.join(PROJ, "results","charts","model_performance","all_subs",'sensor')
results_allsubs_slopeint_dir = os.path.join(PROJ, "results","charts","model_performance","all_subs",'slopeint')

os.makedirs(results_allsubs_ANOVA_dir, exist_ok=True)
os.makedirs(results_allsubs_all_dir, exist_ok=True)
os.makedirs(results_allsubs_corr_dir, exist_ok=True)
os.makedirs(results_allsubs_LME10_dir, exist_ok=True)
os.makedirs(results_allsubs_sensor_dir, exist_ok=True)
os.makedirs(results_allsubs_slopeint_dir, exist_ok=True)

# Specific data files
V1_DAY_RAW        = os.path.join(DATA, "v1_day_raw.csv")
V1_DAY_IMPUTED    = os.path.join(DATA, "v1_day_trainval_imputed.csv")
V2_WEEK_IMPUTED   = os.path.join(DATA, "v2_week_trainval_imputed.csv")




