# src/paths.py
from pathlib import Path
import os

# Whole folder. Assuming paths.py is in a subfolder
PROJ = os.path.join(os.getcwd(), "..")

# Project folders
DATA_FOLDER = os.path.join(PROJ, "data")
SCRIPTS = os.path.join(PROJ, "data")
NB = os.path.join(PROJ, "data")
PIPE = os.path.join(PROJ, "pipeline")
RESULTS = os.path.join(PROJ, "results")

# Data folders
RAW = os.path.join(DATA_FOLDER, "raw")
DATA = os.path.join(DATA_FOLDER, "dfs")
SUB_DFS = os.path.join(DATA_FOLDER, "sub_dfs")

# Specific data files
V1_DAY_RAW        = os.path.join(DATA, "v1_day_raw.csv")
V1_DAY_IMPUTED    = os.path.join(DATA, "v1_day_trainval_imputed.csv")
V2_WEEK_IMPUTED   = os.path.join(DATA, "v2_week_trainval_imputed.csv")




