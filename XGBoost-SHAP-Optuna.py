#!/usr/bin/env python3
import os
import sys
import warnings
import fnmatch
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.samplers import TPESampler
from matplotlib.ticker import MultipleLocator # Required import

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'xgboost_feature_matrix_final.csv'
TARGET_COL = 'rel_en_absolute'
#TARGET_COL = 'rel_en_per_atom'
GROUP_COL = 'system'
N_TRIALS = 50

TRAINING_EXCLUSIONS = [
    'system', 
    'isomer', 
    'tot_en', 
    'n_atoms', 
    'Has_Anomaly',
    'rel_en_per_atom',
    'rel_en_absolute',
    'z_score_energy',
    'I_*',
    '*_Count',
    'has_*',
    'Mes_Centroid_Dist_Mean',
    'Cps_Centroid_Dist_Mean',
    'Asymmetry_Parameter',
    'Zn_ECN_Avg',
    'Global_ECN_Avg',
    'Global_ECN_Var',
    'homo',
    'lumo',
    '*_mu',
    'Mes_Nearest_Core_Dist_Mean',
    'total_core_charge',
    '*_elec',
    'gap',
    'Cu_ECN_Avg',
    '*_max_q',
    '*_min_q'
]

SHAP_EXCLUSIONS = ['CO2_Nearest_Core_Dist_Mean', 'Cps_R3_mean']

PUBLISH_LABELS = {
    'Mean_Dist_Cps_to_Nearest_Zn': r'$d(\mathrm{Cp}^*_{\mathrm{centroid}}-\mathrm{Zn}_{\mathrm{nearest}})$',
    'Mean_Dist_Cps_to_Nearest_Cu': r'$d(\mathrm{Cp}^*_{\mathrm{centroid}}-\mathrm{Cu}_{\mathrm{nearest}})$',
    'Mean_Dist_Mes_to_Nearest_Zn': r'$d(\mathrm{Mes}_{\mathrm{centroid}}-\mathrm{Zn}_{\mathrm{nearest}})$',
    'Mean_Dist_Mes_to_Nearest_Cu': r'$d(\mathrm{Mes}_{\mathrm{centroid}}-\mathrm{Cu}_{\mathrm{nearest}})$',
    'Cu_ECN_Var': r'$\sigma^2(\mathrm{ECN}_{\mathrm{Cu}})$',
    'Zn_ECN_Var': r'$\sigma^2(\mathrm{ECN}_{\mathrm{Zn}})$',
    'Core_Density': r'$\rho_{\mathrm{core}}$',
    'Core_Bond_Variance': r'$\sigma^2(b_{\mathrm{core}})$',
    'Cps_R3_mean': r'$\mathrm{Cp^*}_{\mu _3}$',
    'Mes_R2_mean': r'$\mathrm{Mes}_{\mu _2}$',
    'S_sigma_Measure': r'$S_\sigma$',
    'Cps_R2_mean': r'$\mathrm{Cp^*}_{\mu _2}$',
    'NPR2': r'NPR$_2$',
    'Cu_rel_s_center': r'$\Delta$Cu$_{s-center}$',
    'Zn_rel_d_center': r'$\Delta$Cu$_{d-center}$',
    'Zn_mean_q': r'$\bar{q}_{\mathrm{Zn}}$',
    'Cu_var_d': r'$\sigma^2(d_{\mathrm{Cu}})$',
    'Zn_delta_q': r'$\Delta$q$_\mathrm{Zn}$',
    'Zn_var_q': r'$\sigma^2(q_{\mathrm{Zn}})$'
}
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model and generate SHAP plots.")
    parser.add_argument('--min_system', type=int, default=5, help="Minimum system number to include.")
    parser.add_argument('--max_system', type=int, default=24, help="Maximum system number to include.")
    args = parser.parse_args()

    min_sys = args.min_system
    max_sys = args.max_system

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        sys.exit(1)

    output_folder='figures-XGBoost-elec'
    os.makedirs(f'./{output_folder}', exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    
    if TARGET_COL not in df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found.")
        sys.exit(1)

    initial_count = len(df)
    
    system_nums = df[GROUP_COL].astype(str).str.extract(r'(\d+)')[0].astype(float)
    mask = (system_nums >= min_sys) & (system_nums <= max_sys)
    df = df[mask].copy()
    
    filtered_count = len(df)
    print(f"System filter ({min_sys} <= system <= {max_sys}) applied.")
    print(f"Dataset reduced from {initial_count} to {filtered_count} viable isomers.")
    
    if filtered_count == 0:
        print("Error: Filtered dataset is empty. Check your system range limits.")
        sys.exit(1)

    drop_cols = []
    for pattern in TRAINING_EXCLUSIONS:
        drop_cols.extend(fnmatch.filter(df.columns, pattern))
    drop_cols = list(set(drop_cols)) 
    
    if TARGET_COL not in drop_cols:
        drop_cols.append(TARGET_COL)

    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features].copy()
    y = df[TARGET_COL].copy()

    print(f"Features selected for training: {len(features)}")
    print(f"Starting Optuna optimization ({N_TRIALS} trials)...")

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'random_state': 42,
            'n_jobs': -1
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(X))
        
        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            model = xgb.XGBRegressor(**params, n_estimators=1000, early_stopping_rounds=20)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            oof_preds[test_idx] = model.predict(X_te)
            
        return np.sqrt(mean_squared_error(y, oof_preds))

    sampler = TPESampler(multivariate=True, n_startup_trials=30, seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({'random_state': 42, 'n_jobs': -1})

    print("-" * 30)
    print(f"Best OOF RMSE: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in best_params.items():
        if k not in ['random_state', 'n_jobs']:
            print(f"  {k}: {v}")
    print("-" * 30)

    print("Executing 5-Fold Cross Validation for SHAP extraction...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X))
    oof_shap_values = np.zeros(X.shape)
    best_iterations = []
    
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBRegressor(**best_params, n_estimators=1000, early_stopping_rounds=20)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        
        best_iterations.append(model.best_iteration)
        oof_predictions[test_idx] = model.predict(X_te)
        
        explainer = shap.TreeExplainer(model)
        oof_shap_values[test_idx] = explainer.shap_values(X_te)

    overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    overall_r2 = r2_score(y, oof_predictions)

    print("Generating Parity Plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y, oof_predictions, alpha=0.6, edgecolor='black', zorder=2)

    axis_min = min(y.min(), oof_predictions.min())
    axis_max = max(y.max(), oof_predictions.max())
    padding = (axis_max - axis_min) * 0.05

    plt.plot([axis_min - padding, axis_max + padding], 
             [axis_min - padding, axis_max + padding], 
             'r--', lw=2, zorder=1, label='Perfect Fit')

    plt.xlim(axis_min - padding, axis_max + padding)
    plt.ylim(axis_min - padding, axis_max + padding)
    plt.xlabel('DFT Relative Energy (eV/atom)', fontsize=12)
    plt.ylabel('Predicted Relative Energy (eV/atom)', fontsize=12)

    # --- New Tick Configuration ---
    ax = plt.gca()

    # Set major and minor tick intervals
    ax.xaxis.set_major_locator(MultipleLocator(0.50))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.50))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

    # Set ticks to point inwards for both axes and both major/minor ticks
    # Added top=True and right=True as inward ticks usually span the full bounding box in scientific plots
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    # ------------------------------

    metrics_text = f"OOF RMSE: {overall_rmse:.4f}\nOOF R²: {overall_r2:.4f}"
    plt.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.legend(loc='lower right')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'./{output_folder}/parity_plot.png', dpi=300)
    plt.close()

    print("Generating SHAP plots...")
    valid_targets = [col for col in X.columns if col not in SHAP_EXCLUSIONS]
    target_indices = [X.columns.get_loc(col) for col in valid_targets]
    
    shap_values_filtered = oof_shap_values[:, target_indices]
    X_sample_filtered = X.iloc[:, target_indices]
    
    X_plot = X_sample_filtered.rename(columns=PUBLISH_LABELS)
    
    # Export arrays for standalone plotting
    np.save(f'./{output_folder}/shap_values_filtered.npy', shap_values_filtered)
    X_plot.to_pickle(f'./{output_folder}/X_plot.pkl')

    figure_height = max(8, len(valid_targets) * 0.3)
    
    plt.figure(figsize=(10, figure_height))
    shap.summary_plot(shap_values_filtered, X_plot, max_display=15, show=False)
    plt.xlabel(r"SHAP value (impact on $\Delta E$)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'./{output_folder}/shap_beeswarm.pdf', dpi=300)
    plt.close()

    plt.figure(figsize=(10, figure_height))
    shap.summary_plot(shap_values_filtered, X_plot, plot_type="bar", max_display=len(valid_targets), show=False)
    plt.xlabel(r"mean(|SHAP value|) (average impact on $\Delta E$)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'./{output_folder}/shap_bar.pdf', dpi=300)
    plt.close()

    mean_abs_shap = np.abs(shap_values_filtered).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:4]
    top_features = X_sample_filtered.columns[top_indices]

    for feature in top_features:
        plot_feature_name = PUBLISH_LABELS.get(feature, feature)
        shap.dependence_plot(plot_feature_name, shap_values_filtered, X_plot, show=False)
        plt.tight_layout()
        plt.savefig(f'./{output_folder}/shap_scatter_{feature}.pdf', dpi=300)
        plt.close()

    print("Training final model and extracting importances...")
    avg_best_iteration = int(np.mean(best_iterations))
    print(f"Average optimal trees from CV: {avg_best_iteration}")
    
    best_params['n_estimators'] = avg_best_iteration
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X, y)
    final_model.save_model('xgboost_isomer_model.json')
    
    importance_weight = final_model.get_booster().get_score(importance_type='weight')
    importance_gain = final_model.get_booster().get_score(importance_type='gain')
    importance_cover = final_model.get_booster().get_score(importance_type='cover')
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Weight': [importance_weight.get(f, 0) for f in features],
        'Gain': [importance_gain.get(f, 0) for f in features],
        'Cover': [importance_cover.get(f, 0) for f in features]
    }).sort_values(by='Gain', ascending=False)
    
    importance_df.to_csv('xgboost_feature_importances.csv', index=False)
    print("Complete.")

if __name__ == '__main__':
    main()