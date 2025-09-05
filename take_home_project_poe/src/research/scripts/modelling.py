# mlr_cv_with_diagnostics.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List, Iterable
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.linear_model import HuberRegressor, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt
from scipy import stats

import shap
import xgboost as xgb


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
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
    return arr


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ======================================================================
# Readiness checks (X must already be prepared upstream)
# ======================================================================
def _check_X_is_prepared(X: pd.DataFrame, where: str = "") -> None:
    """
    Lightweight check that the design matrix is already prepared:
      - all columns numeric
      - no NaN/inf
    Prints warnings but does not mutate X.
    """
    where = f" {where}" if where else ""
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        print(f"[warn]{where}: Non-numeric columns in X: {len(non_numeric)} "
              f"(e.g., {non_numeric[:5]}) – did you forget prepare_design_matrix?")

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        print(f"[warn]{where}: NaNs present in X: {len(bad)} cols (e.g., {bad[:5]})")

    num = X.select_dtypes(include=[np.number])
    if np.isinf(num).any().any():
        bad = num.columns[np.isinf(num).any()].tolist()
        print(f"[warn]{where}: Infs present in X: {len(bad)} cols (e.g., {bad[:5]})")


# ======================================================================
# Helpers for models inside Pipelines
# ======================================================================
def _get_linear_step(model_or_pipe):
    """Return the linear estimator (HuberRegressor/SGDRegressor) from a Pipeline or the estimator itself."""
    if hasattr(model_or_pipe, "coef_"):
        return model_or_pipe
    if hasattr(model_or_pipe, "named_steps"):
        for name in reversed(list(model_or_pipe.named_steps.keys())):
            est = model_or_pipe.named_steps[name]
            if hasattr(est, "coef_"):
                return est
    return None


def _coef_frame(model_or_pipe, columns: List[str]) -> pd.DataFrame:
    est = _get_linear_step(model_or_pipe)
    if est is None or not hasattr(est, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
    coef = np.asarray(getattr(est, "coef_")).ravel()
    df = pd.DataFrame({"feature": columns, "coefficient": coef})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False)


def _best_cv_mean_std(gs: GridSearchCV) -> Tuple[float, float]:
    idx = gs.best_index_
    means = gs.cv_results_["mean_test_score"]
    stds = gs.cv_results_["std_test_score"]
    return float(means[idx]), float(stds[idx])


# ======================================================================
# Cross-validation (uses Pipeline with StandardScaler)
# ======================================================================
def cross_val_r2_huber(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    epsilon: float = 1.35,   # Huber transition point
    alpha: float = 0.0,      # L2 shrinkage; 0.0 ≈ unregularized
    max_iter: int = 5000,
    tol: float = 1e-5,
    verbose: bool = True,
) -> List[float]:
    """K-fold R^2 using Huber loss (robust linear) with a Pipeline (scaler + Huber)."""
    _check_X_is_prepared(X, where="[CV]")
    y_vec = _ensure_vector(y)
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    scores: List[float] = []
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y_vec[tr], y_vec[va]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("huber", HuberRegressor(
                epsilon=epsilon, alpha=alpha,
                fit_intercept=True, max_iter=max_iter, tol=tol
            )),
        ])
        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_va)
        r2 = r2_score(y_va, y_hat)
        scores.append(float(r2))
        if verbose:
            print(f"[Fold {fold}/{n_splits}] R² = {r2:.4f}")

    if verbose:
        print(f"CV R²: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
    return scores


# ======================================================================
# Training + holdout diagnostics (Pipeline with scaler)
# ======================================================================
def train_mlr_with_holdout(  # kept name for compatibility
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    epsilon: float = 1.35,
    alpha: float = 0.0,       # 0.0 ≈ no ridge
    max_iter: int = 5000,
    tol: float = 1e-5,
    verbose: bool = True,
    plots: bool = True,
    model_name: str = "Huber MLR",
) -> Dict[str, Any]:
    """Train/test split and fit robust linear (Huber loss) using Pipeline(scaler+Huber)."""
    _check_X_is_prepared(X, where="[train/test]")
    y_vec = _ensure_vector(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_vec, test_size=test_size, random_state=random_state
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("huber", HuberRegressor(
            epsilon=epsilon, alpha=alpha,
            fit_intercept=True, max_iter=max_iter, tol=tol
        )),
    ])
    model.fit(X_train, y_train)

    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)

    metrics = {
        "r2_train": float(r2_score(y_train, y_pred_tr)),
        "r2_test": float(r2_score(y_test, y_pred_te)),
        "rmse_train": _rmse(y_train, y_pred_tr),
        "rmse_test": _rmse(y_test, y_pred_te),
        "mae_train": float(mean_absolute_error(y_train, y_pred_tr)),
        "mae_test": float(mean_absolute_error(y_test, y_pred_te)),
    }
    if verbose:
        print(f"\n[{model_name}] Holdout metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    coef_df = _coef_frame(model, X_train.columns)

    if plots:
        plot_basic_residuals(y_train, y_pred_tr, y_test, y_pred_te, model_name=model_name)

    return {
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred_train": y_pred_tr, "y_pred_test": y_pred_te,
        "metrics": metrics,
        "coefficients": coef_df,
    }


# ======================================================================
# Residual diagnostics (focused on requested metrics)
# ======================================================================
def plot_basic_residuals(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    *,
    model_name: str = "Model",
) -> None:
    """2×2 residual diagnostic plots (matplotlib only)."""
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{model_name}: Basic Residual Diagnostics", fontsize=15)

    # 1. Residuals vs Fitted (Train)
    axes[0, 0].scatter(y_pred_train, residuals_train, alpha=0.6)
    axes[0, 0].axhline(0, color="r", linestyle="--")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("1. Residuals vs Fitted (Training)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Q-Q plot (Train residuals)
    stats.probplot(residuals_train, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("2. Q-Q Plot (Training)")

    # 3. Residuals vs Fitted (Test)
    axes[1, 0].scatter(y_pred_test, residuals_test, alpha=0.6, color="orange")
    axes[1, 0].axhline(0, color="r", linestyle="--")
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("3. Residuals vs Fitted (Test)")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Actual vs Predicted (Test)
    axes[1, 1].scatter(y_test, y_pred_test, alpha=0.6, color="orange")
    lo = min(np.min(y_test), np.min(y_pred_test))
    hi = max(np.max(y_test), np.max(y_pred_test))
    axes[1, 1].plot([lo, hi], [lo, hi], "r--", lw=2)
    axes[1, 1].set_xlabel("Actual")
    axes[1, 1].set_ylabel("Predicted")
    axes[1, 1].set_title("4. Actual vs Predicted (Test)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def analyze_linear_performance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray | pd.DataFrame,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series | np.ndarray | pd.DataFrame] = None,
    metadata: Optional[pd.DataFrame] = None,
    *,
    model_name: Optional[str] = None,
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
    header = "PERFORMANCE DIAGNOSTICS"
    if model_name:
        header = f"{model_name} {header}"
    print(header)
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
    for col in [
        "explicit_mod_count", "implicit_mod_count",
        "pdps", "fdps", "cdps", "ldps", "chaos_dps",
        "req_level", "req_str", "req_dex", "req_int",
        "ilvl", "quality", "socket_count", "open_slots_est"
    ]:
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
    else:
        df["rarity"] = "rarity_unknown"

    # total sockets
    if "socket_count" in df.columns:
        df["total_sockets"] = df["socket_count"]
    else:
        socket_cols = [c for c in X_test.columns if "socket" in str(c).lower()]
        if socket_cols:
            df["total_sockets"] = X_test[socket_cols].sum(axis=1)

    # corrupted flag (from X or metadata)
    corrupt_col = None
    for c in X_test.columns:
        if str(c).lower() in {"corrupted", "is_corrupted"}:
            corrupt_col = c
            break
    if corrupt_col is not None:
        df["corrupted"] = X_test[corrupt_col].astype(bool)
    elif metadata is not None:
        for c in metadata.columns:
            if str(c).lower() in {"corrupted", "is_corrupted"}:
                df["corrupted"] = metadata.loc[df.index, c].astype(bool)
                break

    return df


def plot_residual_patterns(
    analysis_df: pd.DataFrame,
    *,
    model_name: str = "Model",
) -> None:
    """
    Focused residual panels (dynamic grid). Always shows:
      - Residuals vs Predicted
      - Actual vs Predicted
    And, if present in analysis_df, also shows:
      explicit_mod_count, implicit_mod_count,
      pdps, fdps, cdps, ldps, chaos_dps,
      req_level, req_str, req_dex, req_int,
      ilvl,
      Residual distribution (hist),
      Residuals by Socket Count (boxplot),
      Residuals by Corruption (bar),
      Residuals by Rarity (bar)
    """
    panels: List[Tuple[str, Optional[str]]] = []

    # Always include these first/last
    panels.append(("Residuals vs Predicted", None))

    # Scatter panels for requested numeric features
    feature_order = [
        "explicit_mod_count", "implicit_mod_count",
        "pdps", "fdps", "cdps", "ldps", "chaos_dps",
        "req_level", "req_str", "req_dex", "req_int",
        "ilvl",
    ]
    for f in feature_order:
        if f in analysis_df.columns:
            panels.append((f, f))

    # Distribution of residuals
    panels.append(("Residual Distribution", None))

    # Socket count boxplot if available
    if "total_sockets" in analysis_df.columns or "socket_count" in analysis_df.columns:
        panels.append(("Residuals by Socket Count", None))

    # Corruption panel if available
    if "corrupted" in analysis_df.columns:
        panels.append(("Residuals by Corruption", None))

    # Rarity panel if available
    if "rarity" in analysis_df.columns and analysis_df["rarity"].notna().any():
        panels.append(("Residuals by Rarity", "rarity"))

    # Always end with actual vs predicted
    panels.append(("Actual vs Predicted", None))

    # Layout
    n = len(panels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.8 * nrows))
    fig.suptitle(f"{model_name}: Residual Pattern Panels", fontsize=15)
    axes = np.atleast_1d(axes).ravel()

    # Helper to safely scatter with a provided title
    def _scatter(ax, x, y, xlabel, panel_title, ylabel="Residuals"):
        if len(x) >= 3:
            ax.scatter(x, y, alpha=0.6)
            ax.axhline(0, color="red", linestyle="--", alpha=0.7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(panel_title)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(panel_title)

    # Plot panels
    for i, (label, feature) in enumerate(panels):
        ax = axes[i]
        prefix = f"{i+1}. "

        if label == "Residuals vs Predicted":
            _scatter(
                ax,
                analysis_df["predicted"], analysis_df["residual"],
                "Predicted Log Price",
                panel_title=prefix + "Residuals vs Predicted"
            )

        elif label == "Actual vs Predicted":
            title = prefix + "Actual vs Predicted"
            if len(analysis_df) > 0:
                ax.scatter(analysis_df["actual"], analysis_df["predicted"], alpha=0.6)
                lo = analysis_df[["actual", "predicted"]].min().min()
                hi = analysis_df[["actual", "predicted"]].max().max()
                ax.plot([lo, hi], [lo, hi], "r--", alpha=0.7)
                ax.set_xlabel("Actual Log Price")
                ax.set_ylabel("Predicted Log Price")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)

        elif label == "Residual Distribution":
            title = prefix + "Residual Distribution"
            if len(analysis_df) > 0:
                ax.hist(analysis_df["residual"], bins=30, alpha=0.7, density=True)
                ax.axvline(0, color="red", linestyle="--", alpha=0.7)
                ax.set_xlabel("Residuals")
                ax.set_ylabel("Density")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)

        elif label == "Residuals by Socket Count":
            title = prefix + "Residuals by Socket Count"
            ts_col = "total_sockets" if "total_sockets" in analysis_df.columns else "socket_count"
            vals, labels = [], []
            if ts_col in analysis_df.columns:
                for s in sorted(analysis_df[ts_col].dropna().unique()):
                    mask = analysis_df[ts_col] == s
                    if mask.sum() >= 5:
                        vals.append(analysis_df.loc[mask, "residual"].values)
                        labels.append(f"{int(s)} sockets")
            if vals:
                ax.boxplot(vals, labels=labels)
                ax.set_ylabel("Residuals")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.text(0.5, 0.5, "Not enough per-socket groups", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(title)

        elif label == "Residuals by Corruption":
            title = prefix + "Prediction Bias by Corruption"
            try:
                grp = (analysis_df
                       .dropna(subset=["corrupted"])
                       .assign(corrupted=lambda d: d["corrupted"].astype(bool))
                       .groupby("corrupted")["residual"]
                       .agg(["mean", "std", "count"])
                       .reindex([False, True]))
                if grp["count"].fillna(0).sum() > 0:
                    yvals = grp["mean"].fillna(0).values
                    yerr = grp["std"].fillna(0).values
                    idxs = np.arange(len(grp))
                    ax.bar(idxs, yvals, yerr=yerr, capsize=5, alpha=0.7)
                    ax.set_xticks(idxs)
                    ax.set_xticklabels(["Not corrupted", "Corrupted"], rotation=0)
                    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
                    ax.set_ylabel("Mean Residual")
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, "No corruption data", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_title(title)
            except Exception:
                ax.text(0.5, 0.5, "Corruption panel error", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(title)

        elif label == "Residuals by Rarity":
            title = prefix + "Prediction Bias by Rarity"
            try:
                grp = (analysis_df
                       .dropna(subset=["rarity"])
                       .groupby("rarity")["residual"]
                       .agg(["mean", "std", "count"])
                       .sort_values("mean"))
                if len(grp) > 0:
                    yvals = grp["mean"].values
                    yerr = grp["std"].fillna(0.0).values
                    idxs = np.arange(len(grp))
                    ax.bar(idxs, yvals, yerr=yerr, capsize=5, alpha=0.7)
                    ax.set_xticks(idxs)
                    ax.set_xticklabels(grp.index, rotation=45, ha="right")
                    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
                    ax.set_ylabel("Mean Residual")
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, "No rarity data", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_title(title)
            except Exception:
                ax.text(0.5, 0.5, "Rarity panel error", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(title)

        else:
            # Generic scatter: residuals vs selected feature
            x = analysis_df[feature]
            y = analysis_df["residual"]
            # For DPS fields, prefer rows with positive values if available
            if feature in {"pdps", "fdps", "cdps", "ldps", "chaos_dps"}:
                mask = x > 0
                if mask.sum() >= 3:
                    x = x[mask]
                    y = y[mask]
            _scatter(
                ax, x, y,
                xlabel=feature,
                panel_title=prefix + f"Residuals vs {feature}"
            )

    # Clear any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ======================================================================
# End-to-end runner (simple interface)
# ======================================================================
def run_huber(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    *,
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    epsilon: float = 1.35,
    alpha: float = 0.0,
    max_iter: int = 5000,
    tol: float = 1e-5,
    plots: bool = True,
) -> Dict[str, Any]:
    """
    Simplest one-call flow (Pipeline):
      - CV (Huber, scaled)
      - Train/Test split with diagnostics
      - Focused residual panels
    """
    _check_X_is_prepared(X, where="[runner]")
    model_name = "Huber MLR"
    print(f"=== Cross-Validation ({model_name}) ===")
    cv_scores = cross_val_r2_huber(
        X, y, n_splits=cv_splits, random_state=random_state,
        epsilon=epsilon, alpha=alpha, max_iter=max_iter, tol=tol, verbose=True
    )

    print("\n=== Train/Test & Diagnostics ===")
    result = train_mlr_with_holdout(
        X, y,
        test_size=test_size, random_state=random_state,
        epsilon=epsilon, alpha=alpha, max_iter=max_iter, tol=tol,
        verbose=True, plots=plots, model_name=model_name
    )

    # print coefficients
    print("\nTop coefficients (by |value|):")
    print(result["coefficients"].head(30).to_string(index=False))

    # richer analysis DataFrame + focused panels
    analysis_df = analyze_linear_performance(
        result["model"], result["X_test"], result["y_test"],
        X_train=result["X_train"], y_train=result["y_train"],
        metadata=metadata, model_name=model_name,
    )
    if plots:
        plot_residual_patterns(analysis_df, model_name=model_name)

    result["cv_scores"] = cv_scores
    result["analysis_df"] = analysis_df
    return result


# === Regularized models: Huber+Ridge & Huber+ElasticNet (grid search + CV)
def run_huber_ridge(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    *,
    alphas: Optional[Iterable[float]] = None,   # HuberRegressor L2 strengths
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: Optional[int] = None,
    epsilon: float = 1.35,
    max_iter: int = 5000,
    tol: float = 1e-5,
    plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Grid search HuberRegressor(alpha=L2) with CV on TRAIN, evaluate on TEST.
    Uses Pipeline(scaler + huber) to avoid data leakage.
    """
    _check_X_is_prepared(X, where="[ridge grid]")

    y_vec = _ensure_vector(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_vec, test_size=test_size, random_state=random_state
    )

    if alphas is None:
        alphas = np.logspace(-6, 1, 18)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("huber", HuberRegressor(
            epsilon=epsilon, fit_intercept=True, max_iter=max_iter, tol=tol
        )),
    ])
    param_grid = {"huber__alpha": list(alphas)}
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    gs = GridSearchCV(
        pipe, param_grid=param_grid, scoring="r2", cv=cv,
        n_jobs=n_jobs, refit=True, return_train_score=True, verbose=0,
    )
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    cv_mean, cv_std = _best_cv_mean_std(gs)

    y_tr_hat = best.predict(X_tr)
    y_te_hat = best.predict(X_te)

    metrics = {
        "model": "Huber+Ridge (L2) via Pipeline",
        "best_params": gs.best_params_,
        "cv_mean_r2": cv_mean, "cv_std_r2": cv_std,
        "r2_train": float(r2_score(y_tr, y_tr_hat)),
        "r2_test": float(r2_score(y_te, y_te_hat)),
        "rmse_train": _rmse(y_tr, y_tr_hat),
        "rmse_test": _rmse(y_te, y_te_hat),
        "mae_train": float(mean_absolute_error(y_tr, y_tr_hat)),
        "mae_test": float(mean_absolute_error(y_te, y_te_hat)),
    }

    if verbose:
        print("\n=== Huber+Ridge (grid search, Pipeline) ===")
        print(f"Best params: {gs.best_params_} | CV R² mean={cv_mean:.4f} ± {cv_std:.4f}")
        for k in ["r2_train", "r2_test", "rmse_test", "mae_test"]:
            print(f"{k}: {metrics[k]:.4f}")
        coef_df = _coef_frame(best, X_tr.columns)
        print("\nTop coefficients (|value|):")
        print(coef_df.head(25).to_string(index=False))

    if plots:
        plot_basic_residuals(y_tr, y_tr_hat, y_te, y_te_hat, model_name="Huber Ridge")

    analysis_df = analyze_linear_performance(best, X_te, y_te, X_train=X_tr, y_train=y_tr,
                                             metadata=metadata, model_name="Huber Ridge")
    if plots:
        plot_residual_patterns(analysis_df, model_name="Huber Ridge")

    return {
        "name": "HuberRidge",
        "grid_search": gs,
        "best_model": best,
        "best_params": gs.best_params_,
        "cv_mean_r2": cv_mean, "cv_std_r2": cv_std,
        "X_train": X_tr, "y_train": y_tr, "y_pred_train": y_tr_hat,
        "X_test": X_te, "y_test": y_te, "y_pred_test": y_te_hat,
        "metrics": metrics,
        "coefficients": _coef_frame(best, X_tr.columns),
        "analysis_df": analysis_df,
    }


def run_huber_elasticnet(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    *,
    alphas: Optional[Iterable[float]] = None,       # SGDRegressor alpha
    l1_ratios: Optional[Iterable[float]] = None,    # SGDRegressor l1_ratio
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: Optional[int] = None,
    epsilon: float = 1.35,                          # Huber width for the loss
    max_iter: int = 5000,
    tol: float = 1e-3,
    plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Grid search ElasticNet with Huber loss via SGDRegressor in a Pipeline(scaler+sgd)."""
    _check_X_is_prepared(X, where="[enet grid]")

    y_vec = _ensure_vector(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_vec, test_size=test_size, random_state=random_state
    )

    if alphas is None:
        alphas = np.logspace(-6, -2, 10)  # SGD alpha is typically small
    if l1_ratios is None:
        l1_ratios = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd", SGDRegressor(
            loss="huber",
            penalty="elasticnet",
            epsilon=epsilon,
            fit_intercept=True,
            learning_rate="optimal",
            early_stopping=False,   # avoid nested validation inside CV
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            average=False,
        )),
    ])
    param_grid = {"sgd__alpha": list(alphas), "sgd__l1_ratio": list(l1_ratios)}
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    gs = GridSearchCV(
        pipe, param_grid=param_grid, scoring="r2", cv=cv,
        n_jobs=n_jobs, refit=True, return_train_score=True, verbose=0,
    )
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    cv_mean, cv_std = _best_cv_mean_std(gs)

    y_tr_hat = best.predict(X_tr)
    y_te_hat = best.predict(X_te)

    metrics = {
        "model": "Huber+ElasticNet (SGD) via Pipeline",
        "best_params": gs.best_params_,
        "cv_mean_r2": cv_mean, "cv_std_r2": cv_std,
        "r2_train": float(r2_score(y_tr, y_tr_hat)),
        "r2_test": float(r2_score(y_te, y_te_hat)),
        "rmse_train": _rmse(y_tr, y_tr_hat),
        "rmse_test": _rmse(y_te, y_te_hat),
        "mae_train": float(mean_absolute_error(y_tr, y_tr_hat)),
        "mae_test": float(mean_absolute_error(y_te, y_te_hat)),
    }

    if verbose:
        print("\n=== Huber+ElasticNet (grid search, Pipeline) ===")
        print(f"Best params: {gs.best_params_} | CV R² mean={cv_mean:.4f} ± {cv_std:.4f}")
        for k in ["r2_train", "r2_test", "rmse_test", "mae_test"]:
            print(f"{k}: {metrics[k]:.4f}")
        coef_df = _coef_frame(best, X_tr.columns)
        nonzero = (coef_df["coefficient"] != 0).sum()
        print(f"\nNon-zero coefficients: {nonzero}/{len(coef_df)}")
        print("\nTop coefficients (|value|):")
        print(coef_df.head(25).to_string(index=False))

    if plots:
        plot_basic_residuals(y_tr, y_tr_hat, y_te, y_te_hat, model_name="Huber ElasticNet")

    analysis_df = analyze_linear_performance(best, X_te, y_te, X_train=X_tr, y_train=y_tr,
                                             metadata=metadata, model_name="Huber ElasticNet")
    if plots:
        plot_residual_patterns(analysis_df, model_name="Huber ElasticNet")

    return {
        "name": "HuberElasticNet",
        "grid_search": gs,
        "best_model": best,
        "best_params": gs.best_params_,
        "cv_mean_r2": cv_mean, "cv_std_r2": cv_std,
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
    epsilon: float = 1.35,
    plots: bool = True,
) -> Dict[str, Any]:
    """
    Train Ridge and ElasticNet (both Huber-based) via grid search (Pipelines),
    show diagnostics for each, and print a compact comparison table.
    """
    ridge_res = run_huber_ridge(
        X, y, metadata,
        alphas=ridge_alphas,
        cv_splits=cv_splits,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        epsilon=epsilon,
        plots=plots,
        verbose=True,
    )

    enet_res = run_huber_elasticnet(
        X, y, metadata,
        alphas=en_alphas,
        l1_ratios=en_l1_ratios,
        cv_splits=cv_splits,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        epsilon=epsilon,
        plots=plots,
        verbose=True,
    )

    summary = pd.DataFrame([
        {
            "model": "HuberRidge",
            "best_alpha": ridge_res["best_params"]["huber__alpha"],
            "best_l1_ratio": np.nan,
            "cv_mean_r2": ridge_res["cv_mean_r2"],
            "cv_std_r2": ridge_res["cv_std_r2"],
            "test_r2": ridge_res["metrics"]["r2_test"],
            "test_rmse": ridge_res["metrics"]["rmse_test"],
            "test_mae": ridge_res["metrics"]["mae_test"],
        },
        {
            "model": "HuberElasticNet",
            "best_alpha": enet_res["best_params"]["sgd__alpha"],
            "best_l1_ratio": enet_res["best_params"]["sgd__l1_ratio"],
            "cv_mean_r2": enet_res["cv_mean_r2"],
            "cv_std_r2": enet_res["cv_std_r2"],
            "test_r2": enet_res["metrics"]["r2_test"],
            "test_rmse": enet_res["metrics"]["rmse_test"],
            "test_mae": enet_res["metrics"]["mae_test"],
        },
    ])
    print("\n=== Regularized Models (Huber, Pipelines): Comparison ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)))

    return {"Ridge": ridge_res, "ElasticNet": enet_res, "summary": summary}


def run_mlr_cv_and_diagnostics(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    *,
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    epsilon: float = 1.35,
    alpha: float = 0.0,
    max_iter: int = 5000,
    tol: float = 1e-5,
    plots: bool = True,
) -> Dict[str, Any]:
    """Alias to keep older imports working; calls run_huber(...)."""
    return run_huber(
        X, y, metadata,
        cv_splits=cv_splits, test_size=test_size,
        random_state=random_state, epsilon=epsilon,
        alpha=alpha, max_iter=max_iter, tol=tol, plots=plots
    )


# ======================================================================
# XGBoost + SHAP (grid search on TRAIN; no scaling)
# ======================================================================
def run_xgb_shap(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    *,
    base_params: Optional[dict] = None,     # fixed params (not tuned)
    param_grid: Optional[dict] = None,      # grid to tune via CV
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: Optional[int] = None,
    plots: bool = True,
    shap_sample: Optional[int] = 5000,      # cap SHAP rows for speed; None = all
) -> Dict[str, Any]:
    """
    XGBoost + SHAP with hyperparameter tuning:
      1) train/test split
      2) GridSearchCV on TRAIN only
      3) best estimator is refit on TRAIN
      4) evaluate on TEST
      5) compute SHAP on TEST (sampled for speed)

    X must be numeric/clean upstream (no NaNs/Infs; no scaling needed).
    """
    _check_X_is_prepared(X, where="[xgb-grid]")
    y_vec = _ensure_vector(y)

    # Split once; CV happens only on X_tr
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_vec, test_size=test_size, random_state=random_state
    )

    # Fixed defaults (NOT tuned)
    default_base = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",          # use "gpu_hist" if you have a GPU
        random_state=random_state,
        n_estimators=1000,           # may also be in the grid
        n_jobs=n_jobs,
    )
    if base_params:
        default_base.update(base_params)

    # Reasonable starter grid (trim/expand as you like)
    if param_grid is None:
        param_grid = {
            "n_estimators": [400, 800, 1200],
            "learning_rate": [0.05, 0.03],
            "max_depth": [4, 6, 8],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "min_child_weight": [1, 5],
            "reg_lambda": [1.0, 5.0, 10.0],
            "reg_alpha": [0.0, 0.1],
        }

    base = xgb.XGBRegressor(**default_base)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,                 # refits the best model on the entire TRAIN split
        return_train_score=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr)

    # Best tuned model (already refit on X_tr by GridSearchCV)
    best: xgb.XGBRegressor = gs.best_estimator_
    cv_mean, cv_std = _best_cv_mean_std(gs)

    # Evaluate on holdout TEST
    y_tr_hat = best.predict(X_tr)
    y_te_hat = best.predict(X_te)

    metrics = {
        "model": "XGBRegressor (GridSearchCV)",
        "best_params": gs.best_params_,
        "cv_mean_r2": cv_mean,
        "cv_std_r2": cv_std,
        "r2_train": float(r2_score(y_tr, y_tr_hat)),
        "r2_test": float(r2_score(y_te, y_te_hat)),
        "rmse_train": _rmse(y_tr, y_tr_hat),
        "rmse_test": _rmse(y_te, y_te_hat),
        "mae_train": float(mean_absolute_error(y_tr, y_tr_hat)),
        "mae_test": float(mean_absolute_error(y_te, y_te_hat)),
    }

    print("\n=== XGBoost (GridSearchCV on TRAIN) ===")
    print(f"Best params: {gs.best_params_} | CV R² mean={cv_mean:.4f} ± {cv_std:.4f}")
    print(f"R² train={metrics['r2_train']:.4f} | test={metrics['r2_test']:.4f}  "
          f"RMSE test={metrics['rmse_test']:.4f}  MAE test={metrics['mae_test']:.4f}")

    # -------------------- SHAP --------------------
    # Additive contributions in log-price space (matches your target)
    explainer = shap.TreeExplainer(best)
    X_te_explain = X_te
    if (shap_sample is not None) and (len(X_te) > shap_sample):
        X_te_explain = X_te.sample(shap_sample, random_state=random_state)

    shap_values = explainer.shap_values(X_te_explain, check_additivity=False)
    base_value = float(np.asarray(explainer.expected_value).ravel()[0])

    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X_te_explain.index)
    shap_df["__base__"] = base_value
    shap_df["pred_log_price"] = best.predict(X_te_explain)
    shap_df["pred_price"] = np.expm1(shap_df["pred_log_price"])
    shap_df["baseline_price"] = np.expm1(base_value)

    # Global importance
    shap_importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
        .sort_values(ascending=False)
        .rename("mean_abs_phi")
        .to_frame()
    )
    shap_importance["median_phi"] = pd.Series(np.median(shap_values, axis=0), index=X.columns)
    shap_importance["median_lift_pct_(1+P)"] = np.exp(shap_importance["median_phi"]) - 1.0

    if plots:
        try:
            shap.summary_plot(shap_values, X_te_explain, show=True, plot_type="dot")
            shap.summary_plot(shap_values, X_te_explain, show=True, plot_type="bar")
        except Exception as e:
            print(f"[warn] SHAP plotting skipped: {e}")

    # Keep analysis/plots consistent with the linear flows
    model_name = "XGBoost"
    analysis_df = analyze_linear_performance(best, X_te, y_te, X_train=X_tr, y_train=y_tr,
                                             metadata=metadata, model_name=model_name)
    if plots:
        plot_residual_patterns(analysis_df, model_name=model_name)

    return {
        "name": "XGB_SHAP_Grid",
        "grid_search": gs,
        "best_model": best,
        "best_params": gs.best_params_,
        "cv_mean_r2": cv_mean,
        "cv_std_r2": cv_std,
        "X_train": X_tr, "y_train": y_tr, "y_pred_train": y_tr_hat,
        "X_test": X_te, "y_test": y_te, "y_pred_test": y_te_hat,
        "metrics": metrics,
        "shap_values": shap_values,
        "shap_base_value": base_value,
        "shap_df": shap_df,
        "shap_importance": shap_importance,
        "analysis_df": analysis_df,
    }


def decompose_item_with_shap(
    model, explainer, x_row: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return a tidy table for a SINGLE item with:
      - phi (log-space contribution)
      - multiplicative lift on (1+price): exp(phi)
      - running product and predicted price decomposition
    """
    if not isinstance(x_row, (pd.Series, pd.DataFrame)):
        x_row = pd.Series(x_row, index=getattr(model, "feature_names_in_", None))
    x = x_row.to_frame().T

    phi = explainer.shap_values(x, check_additivity=False).ravel()
    base = float(np.asarray(explainer.expected_value).ravel()[0])

    df = pd.DataFrame({
        "feature": x.columns,
        "value": x.iloc[0].values,
        "phi_log": phi,
        "lift_mult_(1+P)": np.exp(phi),
        "lift_pct_(1+P)": np.exp(phi) - 1.0,
    }).sort_values("phi_log", ascending=False)

    pred_log = float(model.predict(x)[0])
    pred_price = float(np.expm1(pred_log))
    baseline_price = float(np.expm1(base))

    # running decomposition (optional, helpful to sanity check additivity)
    df["cum_log"] = base + df["phi_log"].cumsum()
    df["cum_price"] = np.expm1(df["cum_log"])

    summary = pd.DataFrame({
        "baseline_log": [base],
        "pred_log": [pred_log],
        "baseline_price": [baseline_price],
        "pred_price": [pred_price],
    })
    return df, summary


# ======================================================================
# Example usage (commented)
# ======================================================================
# if __name__ == "__main__":
#     # Upstream you should do:
#     #   X_raw, y_df, meta = modelling.load_and_process_item_data(...)
#     #   X, info = prepare_design_matrix(X_raw, standardize=False, ...)  # we scale in Pipelines now
#     #
#     # Then call any of these:
#     # out = run_huber(X, y_df["log_price"], metadata=meta, cv_splits=5, test_size=0.2)
#     # out = run_huber_ridge(X, y_df["log_price"], metadata=meta)
#     # out = run_huber_elasticnet(X, y_df["log_price"], metadata=meta)
#     # xgb_out = run_xgb_shap(X, y_df["log_price"], metadata=meta)
#     # cmp = compare_regularized_models(X, y_df["log_price"], metadata=meta)
