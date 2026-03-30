"""
model.py
--------
Train, evaluate, tune, and save ML models for crop yield prediction.
Models: Linear Regression, Random Forest, XGBoost, LightGBM
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.linear_model   import LinearRegression
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

try:
    from xgboost  import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

ACCENT = "#2d6a4f"
BG     = "#f8fdf4"

# ─────────────────────────────────────────────
# 1. Train / test split
# ─────────────────────────────────────────────
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state)


# ─────────────────────────────────────────────
# 2. Build model zoo
# ─────────────────────────────────────────────
def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.08,
            max_depth=5, random_state=42
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        )
    if LGB_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.05,
            num_leaves=63, random_state=42, n_jobs=-1, verbose=-1
        )
    return models


# ─────────────────────────────────────────────
# 3. Evaluate one model
# ─────────────────────────────────────────────
def evaluate_model(model, X_train, X_test, y_train, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return {
        "model"      : model,
        "y_pred"     : y_pred,
        "MAE"        : round(mae,  2),
        "RMSE"       : round(rmse, 2),
        "R2"         : round(r2,   4),
        "Train Time" : train_time,
    }


# ─────────────────────────────────────────────
# 4. Train all models & return comparison df
# ─────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test):
    models  = get_models()
    results = {}
    for name, mdl in models.items():
        results[name] = evaluate_model(mdl, X_train, X_test, y_train, y_test)
    return results


# ─────────────────────────────────────────────
# 5. Cross validation
# ─────────────────────────────────────────────
def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y,
                             scoring="r2", cv=cv, n_jobs=-1)
    return {
        "cv_mean" : round(scores.mean(), 4),
        "cv_std"  : round(scores.std(),  4),
        "cv_all"  : scores.tolist(),
    }


# ─────────────────────────────────────────────
# 6. Hyperparameter tuning (Random Forest)
# ─────────────────────────────────────────────
def tune_random_forest(X_train, y_train, n_iter=20, cv=3, random_state=42):
    param_dist = {
        "n_estimators" : [100, 200, 300, 500],
        "max_depth"    : [6, 10, 15, 20, None],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf"  : [1, 2, 4],
        "max_features"      : ["sqrt", "log2", 0.5],
    }
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=n_iter, scoring="r2",
        cv=cv, random_state=random_state, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, round(search.best_score_, 4)


# ─────────────────────────────────────────────
# 7. Comparison summary dataframe
# ─────────────────────────────────────────────
def results_to_df(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            "Model"      : name,
            "MAE"        : r["MAE"],
            "RMSE"       : r["RMSE"],
            "R² Score"   : r["R2"],
            "Train Time (s)" : r["Train Time"],
        })
    df = pd.DataFrame(rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 8. Visualisations
# ─────────────────────────────────────────────
def plot_model_comparison(results_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    metrics = ["R² Score", "RMSE", "MAE"]
    colors  = ["#40916c", "#74c69d", "#b7e4c7"]

    for ax, metric, color in zip(axes, metrics, colors):
        ax.set_facecolor(BG)
        order = results_df.sort_values(
            metric, ascending=(metric != "R² Score")
        )
        bars = ax.barh(order["Model"], order[metric], color=color, edgecolor="white")
        ax.set_title(metric, fontsize=13, color=ACCENT)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Model Performance Comparison", fontsize=16,
                 color=ACCENT, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred, model_name="Best Model"):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.scatter(y_test, y_pred, alpha=0.3, color=ACCENT, s=12)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Yield (kg/ha)")
    ax.set_ylabel("Predicted Yield (kg/ha)")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=13, color=ACCENT)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_cols: list, top_n: int = 10):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh(imp.index, imp.values,
            color=sns.color_palette("YlGn", top_n))
    ax.set_title("Feature Importance", fontsize=13, color=ACCENT)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return fig


def plot_residuals(y_test, y_pred, model_name="Model"):
    residuals = np.array(y_test) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)

    axes[0].set_facecolor(BG)
    axes[0].scatter(y_pred, residuals, alpha=0.3, color=ACCENT, s=12)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Yield")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot", fontsize=13, color=ACCENT)

    axes[1].set_facecolor(BG)
    sns.histplot(residuals, bins=50, kde=True, color=ACCENT, ax=axes[1])
    axes[1].set_title("Residual Distribution", fontsize=13, color=ACCENT)

    plt.suptitle(f"Residual Analysis — {model_name}", fontsize=14,
                 color=ACCENT, fontweight="bold")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 9. Save / load model
# ─────────────────────────────────────────────
def save_model(model, encoders, feature_cols, path="models/best_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model, "encoders": encoders, "feature_cols": feature_cols}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_model(path="models/best_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
