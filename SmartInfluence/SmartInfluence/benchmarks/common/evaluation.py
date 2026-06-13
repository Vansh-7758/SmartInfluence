"""
SmartInfluence Benchmarking — Evaluation Module

Standardized evaluation pipeline for all 11 models across all trials.
Produces metrics, confusion matrices, comparison charts, and CSV results.
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

import warnings
import os
warnings.filterwarnings('ignore')

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Models that require feature scaling
_SCALE_REQUIRED = {'SVM (RBF)', 'SVM (Linear)', 'Logistic Regression', 'KNN', 'MLP Neural Network'}


def get_all_models(is_binary=False):
    """
    Return an ordered dict of model_name -> model_instance for all 11 models.
    
    Parameters
    ----------
    is_binary : bool
        If True, use binary-compatible loss for CatBoost.
    """
    catboost_loss = 'Logloss' if is_binary else 'MultiClass'
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.2, reg_alpha=0.5, reg_lambda=1.5,
            eval_metric='mlogloss', random_state=42, verbosity=0
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            random_seed=42, verbose=0, loss_function=catboost_loss
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            random_state=42, n_jobs=-1
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', probability=True, random_state=42
        ),
        'SVM (Linear)': SVC(
            kernel='linear', probability=True, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, solver='lbfgs'
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=12, random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, verbose=-1
        ),
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500,
            random_state=42, early_stopping=True,
            validation_fraction=0.15
        ),
    }
    return models


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, model_name, X_train, X_test, y_train, y_test,
                   le, cv_folds=5):
    """
    Train and evaluate a single model with comprehensive metrics.
    
    Returns
    -------
    dict with all metrics, or None if the model fails entirely.
    """
    result = {'model_name': model_name}
    needs_scaling = model_name in _SCALE_REQUIRED
    n_classes = len(le.classes_)
    is_binary = (n_classes == 2)

    # Scale features if needed
    if needs_scaling:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr = X_train.values if hasattr(X_train, 'values') else X_train
        X_te = X_test.values if hasattr(X_test, 'values') else X_test

    # ── Train ──
    start = time.time()
    try:
        model.fit(X_tr, y_train)
    except Exception as e:
        print(f"    ✗ {model_name} — training FAILED: {e}")
        return None
    train_time = time.time() - start

    # ── Predict ──
    y_pred       = model.predict(X_te)
    y_train_pred = model.predict(X_tr)

    # ── Core metrics ──
    result['accuracy']          = accuracy_score(y_test, y_pred)
    result['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
    result['train_accuracy']    = accuracy_score(y_train, y_train_pred)
    result['train_test_gap']    = result['train_accuracy'] - result['accuracy']

    # ── Precision / Recall / F1 ──
    avg = 'binary' if is_binary else 'macro'
    result['precision_macro']   = precision_score(y_test, y_pred, average=avg, zero_division=0)
    result['recall_macro']      = recall_score(y_test, y_pred, average=avg, zero_division=0)
    result['f1_macro']          = f1_score(y_test, y_pred, average=avg, zero_division=0)

    avg_w = 'binary' if is_binary else 'weighted'
    result['precision_weighted'] = precision_score(y_test, y_pred, average=avg_w, zero_division=0)
    result['recall_weighted']    = recall_score(y_test, y_pred, average=avg_w, zero_division=0)
    result['f1_weighted']        = f1_score(y_test, y_pred, average=avg_w, zero_division=0)

    # ── AUC-ROC ──
    try:
        y_proba = model.predict_proba(X_te)
        if is_binary:
            result['auc_roc'] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            result['auc_roc'] = roc_auc_score(y_test, y_proba,
                                              multi_class='ovr', average='macro')
    except Exception:
        result['auc_roc'] = None

    # ── Cross-validation ──
    try:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        if needs_scaling:
            from sklearn.pipeline import make_pipeline
            import copy
            pipe = make_pipeline(StandardScaler(), copy.deepcopy(model))
            cv_scores = cross_val_score(pipe,
                                        pd.concat([X_train, X_test]),
                                        np.concatenate([y_train, y_test]),
                                        cv=cv, scoring='accuracy')
        else:
            import copy
            cv_scores = cross_val_score(copy.deepcopy(model),
                                        pd.concat([X_train, X_test]),
                                        np.concatenate([y_train, y_test]),
                                        cv=cv, scoring='accuracy')
        result['cv_mean']   = cv_scores.mean()
        result['cv_std']    = cv_scores.std()
        result['cv_scores'] = cv_scores.tolist()
    except Exception:
        result['cv_mean']   = None
        result['cv_std']    = None
        result['cv_scores'] = None

    # ── Confusion matrix & report ──
    result['confusion_matrix']       = confusion_matrix(y_test, y_pred).tolist()
    result['classification_report']  = classification_report(
        y_test, y_pred, target_names=le.classes_, zero_division=0
    )
    result['training_time_seconds']  = round(train_time, 2)
    result['classes']                = list(le.classes_)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL MODELS FOR A TRIAL
# ─────────────────────────────────────────────────────────────────────────────

def run_all_models(X_train, X_test, y_train, y_test, le, trial_name,
                   output_dir, is_binary=False):
    """
    Run all 11 models, collect results, save CSV and figures.
    
    Parameters
    ----------
    X_train, X_test, y_train, y_test : arrays
        Train/test split data.
    le : LabelEncoder
        Fitted label encoder.
    trial_name : str
        Display name for this trial.
    output_dir : str
        Directory to save results and figures.
    is_binary : bool
        Whether this is a binary classification trial.
        
    Returns
    -------
    DataFrame with all results.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    models  = get_all_models(is_binary=is_binary)
    results = []

    print(f"\n  Running {len(models)} models for {trial_name}...")
    print(f"  {'─'*60}")

    for i, (name, model) in enumerate(models.items(), 1):
        print(f"  [{i:2d}/{len(models)}] {name}...", end=' ', flush=True)
        
        res = evaluate_model(model, name, X_train, X_test, y_train, y_test,
                             le)
        if res is None:
            print("SKIPPED (error)")
            continue

        results.append(res)
        acc  = res['accuracy'] * 100
        cv   = res['cv_mean'] * 100 if res['cv_mean'] else 0
        gap  = res['train_test_gap'] * 100
        t    = res['training_time_seconds']
        print(f"Acc={acc:.2f}%  CV={cv:.2f}%  Gap={gap:+.2f}%  Time={t:.1f}s")

        # Save confusion matrix figure
        try:
            save_confusion_matrix_figure(
                np.array(res['confusion_matrix']),
                res['classes'], name, trial_name, fig_dir
            )
        except Exception:
            pass

    print(f"  {'─'*60}")

    if not results:
        print("  ✗ No models completed successfully.")
        return pd.DataFrame()

    # Build results DataFrame
    df_results = pd.DataFrame(results)
    cols_to_save = [
        'model_name', 'accuracy', 'balanced_accuracy', 'train_accuracy',
        'train_test_gap', 'cv_mean', 'cv_std',
        'precision_macro', 'recall_macro', 'f1_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted',
        'auc_roc', 'training_time_seconds'
    ]
    cols_to_save = [c for c in cols_to_save if c in df_results.columns]
    df_results[cols_to_save].to_csv(
        os.path.join(output_dir, 'results.csv'), index=False
    )

    # Save comparison chart
    try:
        save_comparison_chart(df_results, trial_name, fig_dir)
    except Exception:
        pass

    # Print summary table
    print(f"\n  {'='*70}")
    print(f"  {trial_name} — RESULTS SUMMARY")
    print(f"  {'='*70}")
    print(f"  {'Model':<22} {'Acc%':>7} {'BalAcc%':>8} {'CV%':>7} {'F1-W':>7} {'AUC':>7} {'Time':>6}")
    print(f"  {'─'*70}")
    for _, row in df_results.sort_values('accuracy', ascending=False).iterrows():
        auc_str = f"{row['auc_roc']:.4f}" if row.get('auc_roc') is not None else '  N/A'
        cv_str  = f"{row['cv_mean']*100:.2f}" if row.get('cv_mean') is not None else '  N/A'
        print(f"  {row['model_name']:<22} "
              f"{row['accuracy']*100:>6.2f}% "
              f"{row['balanced_accuracy']*100:>7.2f}% "
              f"{cv_str:>7} "
              f"{row['f1_weighted']:>6.4f} "
              f"{auc_str:>7} "
              f"{row['training_time_seconds']:>5.1f}s")
    print(f"  {'='*70}")

    # Save per-model classification reports
    reports_path = os.path.join(output_dir, 'classification_reports.txt')
    with open(reports_path, 'w') as f:
        for res in results:
            f.write(f"\n{'='*60}\n")
            f.write(f"Model: {res['model_name']}\n")
            f.write(f"{'='*60}\n")
            f.write(res['classification_report'])
            f.write(f"\nAUC-ROC: {res.get('auc_roc', 'N/A')}\n")

    return df_results


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix_figure(cm, classes, model_name, trial_name, output_dir):
    """Save a confusion matrix heatmap as PNG."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f'{model_name} — {trial_name}', fontsize=13, pad=12)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    plt.tight_layout()

    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    fig.savefig(os.path.join(output_dir, f'cm_{safe_name}.png'), dpi=120)
    plt.close(fig)


def save_comparison_chart(results_df, trial_name, output_dir):
    """Save a bar chart comparing all models' accuracy for this trial."""
    df_sorted = results_df.sort_values('accuracy', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(df_sorted)))
    bars = ax.barh(df_sorted['model_name'], df_sorted['accuracy'] * 100,
                   color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, df_sorted['accuracy'] * 100):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{trial_name} — Model Comparison', fontsize=14, pad=15)
    ax.set_xlim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=120)
    plt.close(fig)
