# Smartphone Sensor Predictive Modeling

This repository contains code and resources for building predictive models using smartphone sensor data. The project encompasses data preprocessing, exploratory data analysis (EDA), feature selection, clustering, and various ML models.

## Folder Structure

The repository is organized into the following directories:

- `/scripts`: Contains Python scripts for various stages of the analysis.
  - `feature_selection.py`: Performs feature selection using linear regression.
  - `modeling.py`: General modeling functions and utilities.
  - `variables.py`: Defines and manages variables used throughout the project.
  - `clustering.py`: Handles clustering of demographic data.
  - `preprocessing.py`: Manages data preprocessing tasks such as merging datasets and handling missing values.
  - `visualization.py`: Provides functions for visualizing data and results.

- `/notebooks`: Contains Jupyter notebooks documenting the analysis process.
  - `01_preprocessing.ipynb`: Data preprocessing steps, including merging datasets and imputing missing values.
  - `02_processing_Pipeline.ipynb`: Pipeline object for value normalization by-column. Transformation of df into wide and slope/intercept versions, for feature extraction. Testing on Hist Gradient Boosting Regressor. Network-based statistic (NBS) for clinical populations vs. healthy population. Hierarchical agglomerative clustering on features + PCA on each cluster. Correlation structures of PCs.
  - `03_demographic_clustering.ipynb`: Correlation across demographic information. Hierarchical agglomerative clustering on demographic features to group subjects into clusters.
   - `03_feature_pca.ipynb`: Correlation across features. Hierarchical agglomerative clustering on demographic features + PCA on each cluster. Correlation structures of extracted PCs. 
  - `04_prediction.ipynb`: Linear regression and nested linear regression of depression score and various covariates. Evaluating six machine learning architectures for prediction of depression score based off passive sensor data. Feature importance analysis of a survey of predictive model types. Visualization of R2 for accuracy of different models for each dataset across different time scales. Comparing imputed vs. nonimputed data.
 

## Analysis Workflow

The project follows these main steps:

1. **Preprocessing**
   - Merging datasets.
   - Imputing missing values.
   - Encoding missingness based on defined thresholds.

2. **Exploratory Data Analysis (EDA)**
   - Visualizing participant data across different weeks.
   - Assessing variable distributions and missingness.
   - Feature selection using linear regression techniques.

3. **Clustering**
   - Variable clustering to identify related features.
   - Demographic data clustering to segment participants.

4. **Visualizing RF Model**
   - Applying Random Forest to build predictive models.
   - Visualiza results

5. **Predictive Modelling using many models**
   - Running a wide variety of models
   - Hyperparameter tuning
   - Evaluating model performance and comparing results.

## Dependencies

To replicate the analysis, pip install the requirements.tct:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Execute the scripts in the `/scripts` directory or run the `01_preprocessing.ipynb` notebook to preprocess the data.

2. **Exploratory Data Analysis**: Use the EDA notebooks (`02_processing_Pipeline.ipynb`, `03_demographic_clustering.ipynb`, etc.) to explore and visualize the data.

3. **Feature Selection and Clustering**: Apply feature selection methods and perform clustering analyses using the corresponding notebooks.

4. **Modeling**: Run the modeling notebooks (`04_prediction.ipynb`, etc.) to build and evaluate predictive models.


---

