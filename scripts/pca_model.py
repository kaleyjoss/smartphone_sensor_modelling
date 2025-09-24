
import os
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import shap
import matplotlib.pyplot as plt


def run_pca_modeling(df_names, regs, brighten_dir, shap_dir, id_columns, n_splits=3):
    id_columns = id_columns + ['idx']
    pca_cv_results = {}
    pca_test_results = {}

    for name in df_names:
        pca_cv_results[name] = {}
        pca_test_results[name] = {}

        # Load training data
        X_train = pd.read_csv(os.path.join(brighten_dir, f'{name}_pca_X_train.csv'))
        y_train = pd.read_csv(os.path.join(brighten_dir, f'{name}_pca_y_train.csv'))
        groups = X_train['num_id']

        # Clean columns
        X_train_cleaned = X_train.drop(columns=[col for col in X_train.columns if col in id_columns or 'Unnamed' in col])
        y_train_cleaned = y_train.drop(columns=[col for col in y_train.columns if 'Unnamed' in col]).squeeze()

        gkf = GroupKFold(n_splits=n_splits)

        for reg_name, reg in regs:
            if 'Decision' in reg_name:
                continue

            print(f'{name}, {reg_name}')
            pca_cv_results[name][reg_name] = {'r2': [], 'mae': [], 'pmc': []}
            pca_test_results[name][reg_name] = {'r2': [], 'mae': [], 'pmc': []}

            for train_idx, val_idx in gkf.split(X_train_cleaned, y_train_cleaned, groups=groups):
                X_tr, X_cv_val = X_train_cleaned.iloc[train_idx], X_train_cleaned.iloc[val_idx]
                y_tr, y_cv_val = y_train_cleaned.iloc[train_idx], y_train_cleaned.iloc[val_idx]

                try:
                    reg.fit(X_tr, y_tr)
                    y_cv_val_pred = reg.predict(X_cv_val)

                    cv_r2 = r2_score(y_cv_val, y_cv_val_pred)
                    cv_mae = mean_absolute_error(y_cv_val, y_cv_val_pred)
                    cv_pmc = pearsonr(y_cv_val, y_cv_val_pred)[0]

                    print(f'CV score: {cv_r2:.3f}, MAE: {cv_mae:.3f}')

                    pca_cv_results[name][reg_name]['r2'].append(cv_r2)
                    pca_cv_results[name][reg_name]['mae'].append(cv_mae)
                    pca_cv_results[name][reg_name]['pmc'].append(cv_pmc)
                except Exception as e:
                    print(f"ERROR: {e}")

            # Load and clean test data
            X_val = pd.read_csv(os.path.join(brighten_dir, f'{name}_pca_X_val.csv'))
            y_val = pd.read_csv(os.path.join(brighten_dir, f'{name}_pca_y_val.csv'))

            X_val_cleaned = X_val.drop(columns=[col for col in X_val.columns if col in id_columns or 'Unnamed' in col])
            y_val_cleaned = y_val.drop(columns=[col for col in y_val.columns if 'Unnamed' in col]).squeeze()

            reg.fit(X_train_cleaned, y_train_cleaned)
            y_val_pred = reg.predict(X_val_cleaned)

            test_r2 = r2_score(y_val_cleaned, y_val_pred)
            test_mae = mean_absolute_error(y_val_cleaned, y_val_pred)
            test_pmc = pearsonr(y_val_cleaned, y_val_pred)[0]

            print(f'Test R²: {test_r2:.3f}, PMC: {test_pmc:.3f}, MAE: {test_mae:.3f}')

            pca_test_results[name][reg_name]['r2'].append(test_r2)
            pca_test_results[name][reg_name]['mae'].append(test_mae)
            pca_test_results[name][reg_name]['pmc'].append(test_pmc)

            if 'Ada' not in reg_name and "agging" not in reg_name:
                try:
                    explainer = shap.Explainer(reg)
                    shap_values = explainer.shap_values(X_val_cleaned)
                    shap.summary_plot(shap_values, X_val_cleaned, show=False)
                    plt.savefig(os.path.join(shap_dir, f"shap_{name}_{reg_name}_pca.png"), bbox_inches="tight", dpi=300)
                    plt.show()
                    plt.clf()
                except Exception as e:
                    print(f"SHAP error: {e}")

    return pca_cv_results, pca_test_results
