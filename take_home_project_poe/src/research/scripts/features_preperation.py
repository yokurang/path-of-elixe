import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any


# ------------------------------------------------------------
# Scaling helpers
# ------------------------------------------------------------
def standardize_features(
    X: pd.DataFrame,
    *,
    fit_stats: Optional[Dict[str, Any]] = None,
    with_centering: bool = True,
    with_scaling: bool = True,
    eps: float = 1e-12,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Z-score standardize numeric features: (x - mean) / std.

    If `fit_stats` is None, compute means/stds from X (training fit).
    If `fit_stats` is provided, apply them (inference transform).

    Non-numeric columns are left unchanged. Boolean columns are cast to float.

    Returns:
        (X_scaled, stats) where stats = {"columns": [...], "means": {...}, "stds": {...}}
    """
    X_out = X.copy()

    # Cast bool → float (keeps them continuous-friendly for linear models)
    for col in X_out.columns:
        if X_out[col].dtype == bool:
            X_out[col] = X_out[col].astype(float)

    numeric_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        if verbose:
            print("No numeric columns to standardize.")
        return X_out, {"columns": [], "means": {}, "stds": {}}

    if fit_stats is None:
        # Fit mode: compute means/std
        means = X_out[numeric_cols].mean(axis=0)
        stds = X_out[numeric_cols].std(axis=0, ddof=0)
        # Avoid divide-by-zero
        stds = stds.mask(stds.abs() < eps, 1.0)

        if with_centering:
            X_out[numeric_cols] = X_out[numeric_cols] - means
        if with_scaling:
            X_out[numeric_cols] = X_out[numeric_cols] / stds

        stats = {
            "columns": numeric_cols,
            "means": means.to_dict(),
            "stds": stds.to_dict(),
            "with_centering": with_centering,
            "with_scaling": with_scaling,
            "eps": eps,
        }
        if verbose:
            print(f"Standardized {len(numeric_cols)} numeric columns (fit).")
        return X_out, stats

    # Transform mode: apply provided stats only to known columns
    cols = fit_stats.get("columns", [])
    means = fit_stats.get("means", {})
    stds = fit_stats.get("stds", {})
    with_centering = fit_stats.get("with_centering", with_centering)
    with_scaling = fit_stats.get("with_scaling", with_scaling)
    eps = fit_stats.get("eps", eps)

    known_cols = [c for c in cols if c in X_out.columns and np.issubdtype(X_out[c].dtype, np.number)]
    unseen_cols = [c for c in X_out.columns if c not in cols and np.issubdtype(X_out[c].dtype, np.number)]

    if with_centering:
        X_out[known_cols] = X_out[known_cols].subtract(pd.Series(means)).astype(float)
    if with_scaling:
        denom = pd.Series(stds).copy()
        # Safety for tiny stds
        denom = denom.mask(pd.Series(denom).abs() < eps, 1.0)
        X_out[known_cols] = X_out[known_cols].divide(denom, axis=1)

    if verbose:
        print(f"Standardized {len(known_cols)} numeric columns (transform).")
        if unseen_cols:
            print(f"Note: {len(unseen_cols)} numeric columns had no scaler stats and were left unscaled.")

    return X_out, fit_stats


def apply_scaler(
    X: pd.DataFrame,
    scaler_stats: Dict[str, Any],
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply previously fitted scaler stats to X (transform only)."""
    X_scaled, _ = standardize_features(
        X,
        fit_stats=scaler_stats,
        with_centering=scaler_stats.get("with_centering", True),
        with_scaling=scaler_stats.get("with_scaling", True),
        eps=scaler_stats.get("eps", 1e-12),
        verbose=verbose,
    )
    return X_scaled


def inverse_standardize(
    X_scaled: pd.DataFrame,
    scaler_stats: Dict[str, Any],
) -> pd.DataFrame:
    """Invert z-score standardization for debugging/inspection."""
    X_inv = X_scaled.copy()
    cols = scaler_stats.get("columns", [])
    means = scaler_stats.get("means", {})
    stds = scaler_stats.get("stds", {})
    with_centering = scaler_stats.get("with_centering", True)
    with_scaling = scaler_stats.get("with_scaling", True)

    known_cols = [c for c in cols if c in X_inv.columns and np.issubdtype(X_inv[c].dtype, np.number)]

    if with_scaling:
        X_inv[known_cols] = X_inv[known_cols].multiply(pd.Series(stds), axis=1)
    if with_centering:
        X_inv[known_cols] = X_inv[known_cols].add(pd.Series(means), axis=1)
    return X_inv


# ------------------------------------------------------------
# Design matrix preparation
# ------------------------------------------------------------
def prepare_design_matrix(
    X: pd.DataFrame,
    *,
    one_hot_prefixes: Tuple[str, ...] = ("rarity_",),
    baseline_preferences: Optional[Dict[str, str]] = None,
    drop_zero_variance: bool = True,
    standardize: bool = True,
    standardize_fit_stats: Optional[Dict[str, Any]] = None,  # None → fit; dict → transform
    with_centering: bool = True,
    with_scaling: bool = True,
    eps: float = 1e-12,
    inplace: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepare feature matrix for modeling.

    Steps:
      1) Drop zero-variance columns (optional)
      2) Resolve dummy-variable trap for specified one-hot families
      3) Cast booleans to float (continuous-friendly)
      4) Standardize numeric features (z-score) if requested

    Args:
        X: Feature matrix
        one_hot_prefixes: Prefixes that identify one-hot families
        baseline_preferences: Preferred baseline to drop per prefix
        drop_zero_variance: Remove zero-variance features before scaling
        standardize: Apply z-score scaling
        standardize_fit_stats: If None, fit scaler on X; else apply provided stats
        with_centering / with_scaling: Control z-score components
        eps: Small epsilon to avoid division by zero
        inplace: Modify input DataFrame in place
        verbose: Print progress

    Returns:
        (cleaned_features, processing_info)
        processing_info includes:
            - zero_variance_dropped
            - baselines_dropped
            - already_baseline_dropped
            - warnings
            - non_numeric_columns
            - standardize: {"columns", "means", "stds", ...}
    """
    features = X if inplace else X.copy()

    if baseline_preferences is None:
        baseline_preferences = {"rarity_": "rarity_normal"}

    processing_info: Dict[str, Any] = {
        "zero_variance_dropped": [],
        "baselines_dropped": [],
        "already_baseline_dropped": [],
        "warnings": [],
        "non_numeric_columns": [],
        "standardize": None,
    }

    # Step 1: Remove zero-variance features
    if drop_zero_variance:
        numeric_features = features.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            variances = numeric_features.var(ddof=0)
            zero_var_features = variances[variances == 0].index.tolist()
            if zero_var_features:
                features.drop(columns=zero_var_features, inplace=True)
                processing_info["zero_variance_dropped"] = zero_var_features
                if verbose:
                    print(f"Dropped {len(zero_var_features)} zero-variance features:")
                    for feature in zero_var_features:
                        print(f"  {feature}")
            elif verbose:
                print("No zero-variance features found")

    # Step 2: Handle dummy variable trap for one-hot encoded families
    for prefix in one_hot_prefixes:
        family_columns = [col for col in features.columns
                          if isinstance(col, str) and col.startswith(prefix)]

        if len(family_columns) <= 1:
            continue  # no trap possible

        family_data = features[family_columns]
        row_sums = family_data.sum(axis=1)

        # Baseline already dropped? (all rows have same < 1.0 sum)
        max_row_sum = row_sums.max()
        if max_row_sum < 1.0 and (row_sums == max_row_sum).all():
            processing_info["already_baseline_dropped"].append(prefix)
            if verbose:
                print(f"Baseline already dropped for '{prefix}' (max row sum: {max_row_sum:.1f})")
            continue

        # Complete 1-of-K → drop one baseline
        if (row_sums == 1).all():
            preferred = baseline_preferences.get(prefix)
            baseline_to_drop = preferred if (preferred in family_columns) else sorted(family_columns)[0]
            features.drop(columns=[baseline_to_drop], inplace=True)
            processing_info["baselines_dropped"].append((prefix, baseline_to_drop))
            if verbose:
                remaining = len(family_columns) - 1
                print(f"Dropped baseline '{baseline_to_drop}' from {prefix} family "
                      f"({remaining} features remaining)")
        elif row_sums.nunique() > 1 and verbose:
            warning = f"Prefix '{prefix}' appears to be partial one-hot (varying row sums)"
            processing_info["warnings"].append(warning)
            print(f"Warning: {warning}")

    # Step 3: Cast booleans to float (so everything is continuous-friendly)
    for col in features.columns:
        if features[col].dtype == bool:
            features[col] = features[col].astype(float)

    # Step 4: Standardize numeric columns (after structure fixes)
    if standardize:
        non_numeric = [c for c in features.columns if not np.issubdtype(features[c].dtype, np.number)]
        if non_numeric:
            processing_info["non_numeric_columns"] = non_numeric
            if verbose:
                print(f"Note: {len(non_numeric)} non-numeric columns left unscaled.")

        features, scaler_stats = standardize_features(
            features,
            fit_stats=standardize_fit_stats,
            with_centering=with_centering,
            with_scaling=with_scaling,
            eps=eps,
            verbose=verbose,
        )
        processing_info["standardize"] = scaler_stats

    if verbose:
        print(f"\nFinal feature count: {features.shape[1]}")

    return features, processing_info


# ------------------------------------------------------------
# Validation & summaries
# ------------------------------------------------------------
def validate_design_matrix(
    X: pd.DataFrame,
    y: Optional[pd.DataFrame] = None,
    *,
    check_multicollinearity: bool = False,
    correlation_threshold: float = 0.99
) -> Dict[str, Any]:
    """
    Validate design matrix for common modeling issues.
    """
    validation = {
        "shape": X.shape,
        "has_missing": False,
        "has_infinite": False,
        "zero_variance_features": [],
        "constant_features": [],
        "highly_correlated_pairs": []
    }

    if X.isna().any().any():
        validation["has_missing"] = True
        missing_cols = X.columns[X.isna().any()].tolist()
        validation["missing_columns"] = missing_cols
        print(f"Warning: {len(missing_cols)} columns have missing values")

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_mask = np.isinf(X[numeric_cols]).any()
        if inf_mask.any():
            validation["has_infinite"] = True
            inf_cols = numeric_cols[inf_mask].tolist()
            validation["infinite_columns"] = inf_cols
            print(f"Warning: {len(inf_cols)} columns have infinite values")

        variances = X[numeric_cols].var(ddof=0)
        zero_var = variances[variances == 0].index.tolist()
        validation["zero_variance_features"] = zero_var
        if zero_var:
            print(f"Warning: {len(zero_var)} features still have zero variance")

    if check_multicollinearity and len(numeric_cols) > 1:
        corr_matrix = X[numeric_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlation = corr_matrix.iloc[i, j]
                if correlation >= correlation_threshold:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, float(correlation)))
        validation["highly_correlated_pairs"] = high_corr_pairs
        if high_corr_pairs and len(high_corr_pairs) <= 10:
            print(f"\nHigh correlations (>{correlation_threshold}):")
            for col1, col2, corr in high_corr_pairs:
                print(f"  {col1} ↔ {col2}: {corr:.3f}")
        elif len(high_corr_pairs) > 10:
            print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs")

    if y is not None and X.shape[0] != y.shape[0]:
        validation["shape_mismatch"] = True
        print(f"Warning: Shape mismatch - X: {X.shape[0]} rows, y: {y.shape[0]} rows")

    return validation


def get_feature_summary(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Get concise summary of feature matrix characteristics.
    """
    summary = {"shape": X.shape}

    feature_types = {}
    for col in X.columns:
        col_str = str(col)
        if col_str.startswith("rarity_"):
            category = "rarity_dummies"
        elif col_str.startswith("sockets_"):
            category = "socket_dummies"
        elif col_str.startswith("cat_"):
            category = "category_dummies"
        elif col_str.endswith("dps"):
            category = "dps_features"
        elif col_str.startswith("req_"):
            category = "requirements"
        elif col_str.startswith("has_") or col_str.startswith("n_"):
            category = "counts_and_flags"
        elif pd.api.types.is_numeric_dtype(X[col]):
            category = "numeric"
        else:
            category = "other"
        feature_types[category] = feature_types.get(category, 0) + 1

    summary["feature_types"] = feature_types

    numeric_features = X.select_dtypes(include=[np.number])
    if not numeric_features.empty:
        summary["numeric_stats"] = {
            "count": len(numeric_features.columns),
            "mean_variance": float(numeric_features.var(ddof=0).mean()),
            "min_value": float(numeric_features.min().min()),
            "max_value": float(numeric_features.max().max())
        }

    return summary


def print_feature_summary(X: pd.DataFrame) -> None:
    """Pretty-print a concise summary of the feature matrix."""
    summary = get_feature_summary(X)
    print(f"Feature Matrix: {summary['shape'][0]:,} samples × {summary['shape'][1]} features")
    print("\nFeature Types:")
    for feature_type, count in sorted(summary["feature_types"].items()):
        print(f"  {feature_type}: {count}")
    if "numeric_stats" in summary:
        stats = summary["numeric_stats"]
        print(f"\nNumeric Features: {stats['count']} total")
        print(f"  Value range: [{stats['min_value']:.2f}, {stats['max_value']:.2f}]")
        print(f"  Mean variance: {stats['mean_variance']:.4f}")
