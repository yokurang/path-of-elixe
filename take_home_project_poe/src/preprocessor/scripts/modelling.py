# mlr_cv_with_diagnostics.py

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List, Iterable
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt
from scipy import stats


# ======================================================================
# Utilities
# ======================================================================
def _ensure_vector(y: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return a 1D numpy array for targets."""
    if isinstance(y, pd.DataFrame):
        if "log_price" in y.columns:
            return y["log_price"].to_numpy()
        if y.shape[1] != 1:
            raise ValueError("y DataFrame must have exactly one column or contain 'log_price'")
        return y.iloc[:, 0].to_numpy()
    if isinstance(y, pd.Series):
        return y.to_numpy()
    return np.asarray(y)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


# ======================================================================
# Cross-validation (assumes X already prepared/standardized upstream)
# ======================================================================
def cross_val_r2_linear(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> List[float]:
    """
    Proper K-fold R^2 for LinearRegression.

    IMPORTANT: X must already be fully prepared (column selection, zero-variance handling,
    and any standardization) BEFORE calling this function. This function does not modify X.
    """
    y_vec = _ensure_vector(y)
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    scores: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_vec[tr_idx], y_vec[va_idx]

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        y_va_pred = model.predict(X_va)
        fold_r2 = r2_score(y_va, y_va_pred)
        scores.append(float(fold_r2))

        if verbose:
            print(f"[Fold {fold}/{n_splits}] R² = {fold_r2:.4f}")

    if verbose:
        print(f"CV R²: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
    return scores


# ======================================================================
# Training + holdout diagnostics (assumes X already prepared upstream)
# ======================================================================
def train_mlr_with_holdout(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train/test split and fit LinearRegression.

    IMPORTANT: X must already be fully prepared/standardized upstream.
    """
    y_vec = _ensure_vector(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_vec, test_size=test_size, random_state=random_state
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_tr = lr.predict(X_train)
    y_pred_te = lr.predict(X_test)

    metrics = {
        "r2_train": float(r2_score(y_train, y_pred_tr)),
        "r2_test": float(r2_score(y_test, y_pred_te)),
        "rmse_train": _rmse(y_train, y_pred_tr),
        "rmse_test": _rmse(y_test, y_pred_te),
        "mae_train": float(mean_absolute_error(y_train, y_pred_tr)),
        "mae_test": float(mean_absolute_error(y_test, y_pred_te)),
        "mape_train": _mape(y_train, y_pred_tr),
        "mape_test": _mape(y_test, y_pred_te),
    }
    if verbose:
        print("\nHoldout metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    coef_df = pd.DataFrame(
        {"feature": X_train.columns, "coefficient": lr.coef_}
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df.sort_values("abs_coefficient", ascending=False, inplace=True)

    return {
        "model": lr,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_tr,
        "y_pred_test": y_pred_te,
        "metrics": metrics,
        "coefficients": coef_df,
    }


# ======================================================================
# Residual diagnostics (2×2 and richer 3×3 set)
# ======================================================================
def plot_basic_residuals(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> None:
    """2×2 residual diagnostic plots (matplotlib only)."""
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Fitted (Train)
    axes[0, 0].scatter(y_pred_train, residuals_train, alpha=0.6)
    axes[0, 0].axhline(0, color="r", linestyle="--")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted (Training)")
    axes[0, 0].grid(True, alpha=0.3)

    # Q-Q plot (Train residuals)
    stats.probplot(residuals_train, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot (Training)")

    # Residuals vs Fitted (Test)
    axes[1, 0].scatter(y_pred_test, residuals_test, alpha=0.6, color="orange")
    axes[1, 0].axhline(0, color="r", linestyle="--")
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("Residuals vs Fitted (Test)")
    axes[1, 0].grid(True, alpha=0.3)

    # Actual vs Predicted (Test)
    axes[1, 1].scatter(y_test, y_pred_test, alpha=0.6, color="orange")
    lo = min(np.min(y_test), np.min(y_pred_test))
    hi = max(np.max(y_test), np.max(y_pred_test))
    axes[1, 1].plot([lo, hi], [lo, hi], "r--", lw=2)
    axes[1, 1].set_xlabel("Actual")
    axes[1, 1].set_ylabel("Predicted")
    axes[1, 1].set_title("Actual vs Predicted (Test)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_linear_performance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray | pd.DataFrame,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series | np.ndarray | pd.DataFrame] = None,
    metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build an analysis dataframe with residuals, prices (log & expm1), and
    commonly useful feature columns for slicing. Assumes X_test already prepared.
    """
    # Handle DataFrame targets
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test["log_price"] if "log_price" in y_test.columns else y_test.iloc[:, -1]
    if y_train is not None and isinstance(y_train, pd.DataFrame):
        y_train = y_train["log_price"] if "log_price" in y_train.columns else y_train.iloc[:, -1]

    y_pred = model.predict(X_test)
    residuals = _ensure_vector(y_test) - y_pred
    abs_residuals = np.abs(residuals)

    print("=" * 70)
    print("LINEAR REGRESSION PERFORMANCE DIAGNOSTICS")
    print("=" * 70)
    print(f"Overall Test R²: {r2_score(y_test, y_pred):.4f}")
    print(f"Mean Absolute Error: {abs_residuals.mean():.4f}")
    print(f"Root Mean Squared Error: {np.sqrt((residuals**2).mean()):.4f}")

    df = pd.DataFrame(
        {
            "actual": _ensure_vector(y_test),
            "predicted": y_pred,
            "residual": residuals,
            "abs_residual": abs_residuals,
            "actual_price": np.expm1(_ensure_vector(y_test)),
            "predicted_price": np.expm1(y_pred),
        },
        index=X_test.index,
    )

    # Add common features for slicing/plots if present
    for col in ["ilvl", "pdps", "fdps", "cdps", "ldps", "chaos_dps", "crit_chance", "quality",
                "req_level", "req_str", "req_dex", "req_int", "open_slots_est", "socket_count"]:
        if col in X_test.columns:
            df[col] = X_test[col]

    # Rarity: prefer OHE if present, otherwise use metadata if available
    rarity_cols = [c for c in X_test.columns if isinstance(c, str) and c.startswith("rarity_")]
    if rarity_cols:
        rdat = X_test[rarity_cols]
        df["rarity"] = rdat.idxmax(axis=1)
        df.loc[rdat.sum(axis=1) == 0, "rarity"] = "rarity_unknown"
    elif metadata is not None and "rarity" in metadata.columns:
        df["rarity"] = metadata.loc[df.index, "rarity"].astype(str)

    # total sockets
    if "socket_count" in df.columns:
        df["total_sockets"] = df["socket_count"]
    else:
        socket_cols = [c for c in X_test.columns if "socket" in str(c).lower()]
        if socket_cols:
            df["total_sockets"] = X_test[socket_cols].sum(axis=1)

    return df


def plot_residual_patterns(analysis_df: pd.DataFrame) -> None:
    """Richer 3×3 panel of residual patterns."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    # 1. Residuals vs Predicted
    axes[0].scatter(analysis_df["predicted"], analysis_df["residual"], alpha=0.6)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Predicted Log Price")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    # 2. Residuals vs Actual
    axes[1].scatter(analysis_df["actual"], analysis_df["residual"], alpha=0.6)
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Actual Log Price")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Actual")
    axes[1].grid(True, alpha=0.3)

    # 3. Abs residuals by price quantile
    try:
        price_bins = pd.qcut(
            analysis_df["actual"], q=5,
            labels=["Very Low", "Low", "Medium", "High", "Very High"], duplicates="drop"
        )
        agg = analysis_df.groupby(price_bins)["abs_residual"].agg(["mean", "std", "count"])
        axes[2].bar(range(len(agg)), agg["mean"], yerr=agg["std"], capsize=5, alpha=0.7)
        axes[2].set_xticks(range(len(agg)))
        axes[2].set_xticklabels(agg.index, rotation=45)
        axes[2].set_ylabel("Mean Absolute Residual")
        axes[2].set_title("Prediction Error by Price Range")
        axes[2].grid(True, alpha=0.3)
    except ValueError:
        axes[2].text(0.5, 0.5, "Not enough unique\nvalues for binning",
                     ha="center", va="center", transform=axes[2].transAxes)

    # 4. Residuals by rarity (if present)
    if "rarity" in analysis_df.columns:
        grp = analysis_df.groupby("rarity")["residual"].agg(["mean", "std", "count"])
        axes[3].bar(range(len(grp)), grp["mean"], yerr=grp["std"], capsize=5, alpha=0.7)
        axes[3].set_xticks(range(len(grp)))
        axes[3].set_xticklabels(grp.index, rotation=45)
        axes[3].set_ylabel("Mean Residual")
        axes[3].set_title("Prediction Bias by Rarity")
        axes[3].axhline(0, color="red", linestyle="--", alpha=0.7)
        axes[3].grid(True, alpha=0.3)

    # 5. Residuals vs PDPS (if present)
    if "pdps" in analysis_df.columns:
        mask = analysis_df["pdps"] > 0
        if mask.sum() > 10:
            axes[4].scatter(analysis_df.loc[mask, "pdps"], analysis_df.loc[mask, "residual"], alpha=0.6)
            axes[4].axhline(0, color="red", linestyle="--", alpha=0.7)
            axes[4].set_xlabel("Physical DPS")
            axes[4].set_ylabel("Residuals")
            axes[4].set_title("Residuals vs Physical DPS")
            axes[4].grid(True, alpha=0.3)

    # 6. Residuals vs Item Level
    if "ilvl" in analysis_df.columns:
        axes[5].scatter(analysis_df["ilvl"], analysis_df["residual"], alpha=0.6)
        axes[5].axhline(0, color="red", linestyle="--", alpha=0.7)
        axes[5].set_xlabel("Item Level")
        axes[5].set_ylabel("Residuals")
        axes[5].set_title("Residuals vs Item Level")
        axes[5].grid(True, alpha=0.3)

    # 7. Residual distribution
    axes[6].hist(analysis_df["residual"], bins=30, alpha=0.7, density=True)
    axes[6].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[6].set_xlabel("Residuals")
    axes[6].set_ylabel("Density")
    axes[6].set_title("Residual Distribution")
    axes[6].grid(True, alpha=0.3)

    # 8. Actual vs Predicted
    axes[7].scatter(analysis_df["actual"], analysis_df["predicted"], alpha=0.6)
    lo = analysis_df[["actual", "predicted"]].min().min()
    hi = analysis_df[["actual", "predicted"]].max().max()
    axes[7].plot([lo, hi], [lo, hi], "r--", alpha=0.7)
    axes[7].set_xlabel("Actual Log Price")
    axes[7].set_ylabel("Predicted Log Price")
    axes[7].set_title("Actual vs Predicted")
    axes[7].grid(True, alpha=0.3)

    # 9. Boxplot residuals by socket count (if present)
    if "total_sockets" in analysis_df.columns:
        vals = []
        labels = []
        for s in sorted(analysis_df["total_sockets"].dropna().unique()):
            mask = analysis_df["total_sockets"] == s
            if mask.sum() >= 5:
                vals.append(analysis_df.loc[mask, "residual"].values)
                labels.append(f"{int(s)} sockets")
        if vals:
            axes[8].boxplot(vals, labels=labels)
            axes[8].set_ylabel("Residuals")
            axes[8].set_title("Residuals by Socket Count")
            axes[8].grid(True, alpha=0.3)
            axes[8].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


# ======================================================================
# End-to-end runner
# ======================================================================
def run_mlr_cv_and_diagnostics(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    *,
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    - K-fold CV (no internal preprocessing)
    - Train/test with diagnostics
    - Plots (2×2 + 3×3)

    IMPORTANT: X must be preprocessed/standardized upstream.
    """
    print("=== Cross-Validation (LinearRegression) ===")
    cv_scores = cross_val_r2_linear(
        X, y, n_splits=cv_splits, random_state=random_state, verbose=True
    )

    print("\n=== Train/Test & Diagnostics ===")
    result = train_mlr_with_holdout(
        X, y, test_size=test_size, random_state=random_state, verbose=True
    )

    # print coefficients
    print("\nTop coefficients (by |value|):")
    print(result["coefficients"].head(30).to_string(index=False))

    # 2×2 plots
    plot_basic_residuals(
        result["y_train"], result["y_pred_train"], result["y_test"], result["y_pred_test"]
    )

    # richer analysis DataFrame + 3×3 plots
    analysis_df = analyze_linear_performance(
        result["model"], result["X_test"], result["y_test"],
        X_train=result["X_train"], y_train=result["y_train"],
        metadata=metadata,
    )
    plot_residual_patterns(analysis_df)

    result["cv_scores"] = cv_scores
    result["analysis_df"] = analysis_df
    return result


# ======================================================================
# Example usage (uncomment to run)
# ======================================================================
# if __name__ == "__main__":
#     # X must already be prepared (e.g., via your design matrix utils) BEFORE this call.
#     from weapon_features import load_and_process_item_data
#     # Build raw features
#     X_raw, y_df, meta = load_and_process_item_data(
#         "training_data/overall/weapon_bow_overall_standard.parquet",
#         compute_dps=True,
#     )
#     # ...your own prepare/standardize step here...
#     out = run_mlr_cv_and_diagnostics(
#         X_raw, y_df["log_price"], metadata=meta, cv_splits=5, test_size=0.2
#     )
#     model = out["model"]


# === Regularized models: Ridge & ElasticNet (grid search + CV) =================

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge, ElasticNet


def _coef_frame(model, columns: List[str]) -> pd.DataFrame:
    coef = getattr(model, "coef_", None)
    if coef is None:
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
    df = pd.DataFrame({"feature": columns, "coefficient": np.asarray(coef).ravel()})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False)


def _best_cv_mean_std(gs: GridSearchCV) -> Tuple[float, float]:
    idx = gs.best_index_
    means = gs.cv_results_["mean_test_score"]
    stds = gs.cv_results_["std_test_score"]
    return float(means[idx]), float(stds[idx])


def train_ridge_gridcv_with_diagnostics(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    *,
    alphas: Optional[Iterable[float]] = None,
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Grid search Ridge(alpha) with K-fold CV on the TRAIN split, evaluate on TEST,
    print metrics, show residual plots, and return a rich result dict.
    Assumes X is already prepared/standardized upstream.
    """
    y_vec = _ensure_vector(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_vec, test_size=test_size, random_state=random_state
    )

    if alphas is None:
        # for standardized features this is a sensible, wide search
        alphas = np.logspace(-4, 3, 24)

    ridge = Ridge(fit_intercept=True, random_state=random_state)
    param_grid = {"alpha": list(alphas)}
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    gs = GridSearchCV(
        ridge,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr)

    best: Ridge = gs.best_estimator_
    cv_mean, cv_std = _best_cv_mean_std(gs)

    # holdout performance
    y_tr_hat = best.predict(X_tr)
    y_te_hat = best.predict(X_te)

    metrics = {
        "model": "Ridge",
        "best_params": gs.best_params_,
        "cv_mean_r2": cv_mean,
        "cv_std_r2": cv_std,
        "r2_train": float(r2_score(y_tr, y_tr_hat)),
        "r2_test": float(r2_score(y_te, y_te_hat)),
        "rmse_train": _rmse(y_tr, y_tr_hat),
        "rmse_test": _rmse(y_te, y_te_hat),
        "mae_train": float(mean_absolute_error(y_tr, y_tr_hat)),
        "mae_test": float(mean_absolute_error(y_te, y_te_hat)),
        "mape_train": _mape(y_tr, y_tr_hat),
        "mape_test": _mape(y_te, y_te_hat),
    }

    if verbose:
        print("\n=== Ridge (grid search) ===")
        print(f"Best params: {gs.best_params_} | CV R² mean={cv_mean:.4f} ± {cv_std:.4f}")
        for k in ["r2_train", "r2_test", "rmse_test", "mae_test", "mape_test"]:
            print(f"{k}: {metrics[k]:.4f}")

        # top coefficients
        coef_df = _coef_frame(best, X_tr.columns)
        print("\nTop Ridge coefficients (|value|):")
        print(coef_df.head(25).to_string(index=False))

    # residuals: 2×2 + 3×3
    plot_basic_residuals(y_tr, y_tr_hat, y_te, y_te_hat)
    analysis_df = analyze_linear_performance(best, X_te, y_te, X_train=X_tr, y_train=y_tr, metadata=metadata)
    plot_residual_patterns(analysis_df)

    return {
        "name": "Ridge",
        "grid_search": gs,
        "best_model": best,
        "best_params": gs.best_params_,
        "cv_mean_r2": cv_mean,
        "cv_std_r2": cv_std,
        "X_train": X_tr, "y_train": y_tr, "y_pred_train": y_tr_hat,
        "X_test": X_te, "y_test": y_te, "y_pred_test": y_te_hat,
        "metrics": metrics,
        "coefficients": _coef_frame(best, X_tr.columns),
        "analysis_df": analysis_df,
    }


def train_elasticnet_gridcv_with_diagnostics(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    *,
    alphas: Optional[Iterable[float]] = None,
    l1_ratios: Optional[Iterable[float]] = None,
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Grid search ElasticNet(alpha, l1_ratio) with K-fold CV on TRAIN split,
    evaluate on TEST, print metrics, show residual plots, and return result dict.
    Assumes X is already prepared/standardized upstream.
    """
    y_vec = _ensure_vector(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_vec, test_size=test_size, random_state=random_state
    )

    if alphas is None:
        alphas = np.logspace(-4, 1, 20)  # EN is more sensitive; narrower range helps
    if l1_ratios is None:
        l1_ratios = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]

    enet = ElasticNet(
        fit_intercept=True,
        max_iter=10000,
        random_state=random_state,
    )
    param_grid = {
        "alpha": list(alphas),
        "l1_ratio": list(l1_ratios),
    }
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    gs = GridSearchCV(
        enet,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr)

    best: ElasticNet = gs.best_estimator_
    cv_mean, cv_std = _best_cv_mean_std(gs)

    # holdout performance
    y_tr_hat = best.predict(X_tr)
    y_te_hat = best.predict(X_te)

    metrics = {
        "model": "ElasticNet",
        "best_params": gs.best_params_,
        "cv_mean_r2": cv_mean,
        "cv_std_r2": cv_std,
        "r2_train": float(r2_score(y_tr, y_tr_hat)),
        "r2_test": float(r2_score(y_te, y_te_hat)),
        "rmse_train": _rmse(y_tr, y_tr_hat),
        "rmse_test": _rmse(y_te, y_te_hat),
        "mae_train": float(mean_absolute_error(y_tr, y_tr_hat)),
        "mae_test": float(mean_absolute_error(y_te, y_te_hat)),
        "mape_train": _mape(y_tr, y_tr_hat),
        "mape_test": _mape(y_te, y_te_hat),
    }

    if verbose:
        print("\n=== ElasticNet (grid search) ===")
        print(f"Best params: {gs.best_params_} | CV R² mean={cv_mean:.4f} ± {cv_std:.4f}")
        for k in ["r2_train", "r2_test", "rmse_test", "mae_test", "mape_test"]:
            print(f"{k}: {metrics[k]:.4f}")

        coef_df = _coef_frame(best, X_tr.columns)
        nonzero = (coef_df["coefficient"] != 0).sum()
        print(f"\nNon-zero coefficients: {nonzero}/{len(coef_df)}")
        print("\nTop ElasticNet coefficients (|value|):")
        print(coef_df.head(25).to_string(index=False))

    # residuals: 2×2 + 3×3
    plot_basic_residuals(y_tr, y_tr_hat, y_te, y_te_hat)
    analysis_df = analyze_linear_performance(best, X_te, y_te, X_train=X_tr, y_train=y_tr, metadata=metadata)
    plot_residual_patterns(analysis_df)

    return {
        "name": "ElasticNet",
        "grid_search": gs,
        "best_model": best,
        "best_params": gs.best_params_,
        "cv_mean_r2": cv_mean,
        "cv_std_r2": cv_std,
        "X_train": X_tr, "y_train": y_tr, "y_pred_train": y_tr_hat,
        "X_test": X_te, "y_test": y_te, "y_pred_test": y_te_hat,
        "metrics": metrics,
        "coefficients": _coef_frame(best, X_tr.columns),
        "analysis_df": analysis_df,
    }


def compare_regularized_models(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    *,
    ridge_alphas: Optional[Iterable[float]] = None,
    en_alphas: Optional[Iterable[float]] = None,
    en_l1_ratios: Optional[Iterable[float]] = None,
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train Ridge and ElasticNet via grid search, print/plot diagnostics for each,
    and print a compact comparison table of CV + holdout stats.
    """
    ridge_res = train_ridge_gridcv_with_diagnostics(
        X, y, metadata,
        alphas=ridge_alphas,
        cv_splits=cv_splits,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=True,
    )

    enet_res = train_elasticnet_gridcv_with_diagnostics(
        X, y, metadata,
        alphas=en_alphas,
        l1_ratios=en_l1_ratios,
        cv_splits=cv_splits,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=True,
    )

    # side-by-side summary
    summary = pd.DataFrame([
        {
            "model": "Ridge",
            "best_alpha": ridge_res["best_params"]["alpha"],
            "best_l1_ratio": np.nan,
            "cv_mean_r2": ridge_res["cv_mean_r2"],
            "cv_std_r2": ridge_res["cv_std_r2"],
            "test_r2": ridge_res["metrics"]["r2_test"],
            "test_rmse": ridge_res["metrics"]["rmse_test"],
            "test_mae": ridge_res["metrics"]["mae_test"],
            "test_mape": ridge_res["metrics"]["mape_test"],
        },
        {
            "model": "ElasticNet",
            "best_alpha": enet_res["best_params"]["alpha"],
            "best_l1_ratio": enet_res["best_params"]["l1_ratio"],
            "cv_mean_r2": enet_res["cv_mean_r2"],
            "cv_std_r2": enet_res["cv_std_r2"],
            "test_r2": enet_res["metrics"]["r2_test"],
            "test_rmse": enet_res["metrics"]["rmse_test"],
            "test_mae": enet_res["metrics"]["mae_test"],
            "test_mape": enet_res["metrics"]["mape_test"],
        },
    ])

    print("\n=== Regularized Models: Comparison ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)))

    return {"Ridge": ridge_res, "ElasticNet": enet_res, "summary": summary}
