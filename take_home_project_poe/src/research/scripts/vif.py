import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(X: pd.DataFrame, vif_threshold: float = 10.0, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Compute VIF for all numeric features and analyze multicollinearity.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input feature matrix
    vif_threshold : float, default=10.0
        Threshold for detecting multicollinearity
    verbose : bool, default=True
        Whether to print analysis details
    
    Returns:
    --------
    tuple[pd.DataFrame, dict]
        - VIF DataFrame with columns: feature, VIF, tolerance (sorted by VIF desc)
        - Analysis dictionary with multicollinearity summary
    """
    Xn = X.copy()
    
    # Convert non-numeric columns to numeric where possible
    non_numeric_cols = []
    for col in Xn.columns:
        if not pd.api.types.is_numeric_dtype(Xn[col]):
            try:
                converted = pd.to_numeric(Xn[col], errors='coerce')
                if not converted.isna().all():
                    Xn[col] = converted
                    non_numeric_cols.append(col)
            except:
                pass
    
    # Select only numeric columns
    numeric_cols = Xn.select_dtypes(include=[np.number]).columns
    Xn = Xn[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    
    if Xn.shape[1] == 0:
        if verbose:
            print("No numeric columns found; cannot compute VIF.")
        return pd.DataFrame(columns=["feature", "VIF", "tolerance"]), {}
    
    # Add constant and compute VIF
    Xd = sm.add_constant(Xn, has_constant='add')
    feature_matrix = Xd.to_numpy(dtype=float)
    feature_names = Xd.columns.tolist()
    
    vif_results = []
    for i, name in enumerate(feature_names):
        if name == "const":
            continue
        try:
            vif_value = variance_inflation_factor(feature_matrix, i)
        except Exception:
            vif_value = np.inf
        
        tolerance = (1.0 / vif_value) if (np.isfinite(vif_value) and vif_value > 0) else np.nan
        vif_results.append((name, float(vif_value), float(tolerance) if np.isfinite(tolerance) else np.nan))
    
    vif_df = pd.DataFrame(vif_results, columns=["feature", "VIF", "tolerance"]).sort_values("VIF", ascending=False)
    
    # Analysis
    high_vif = vif_df[vif_df["VIF"] > vif_threshold]["feature"].tolist()
    moderate_vif = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= vif_threshold)]["feature"].tolist()
    
    analysis = {
        "total_features": len(vif_df),
        "high_multicollinearity_count": len(high_vif),
        "moderate_multicollinearity_count": len(moderate_vif),
        "max_vif": vif_df["VIF"].max() if len(vif_df) > 0 else 0,
        "mean_vif": vif_df["VIF"].mean() if len(vif_df) > 0 else 0,
        "high_vif_features": high_vif,
        "moderate_vif_features": moderate_vif,
        "multicollinearity_detected": len(high_vif) > 0
    }
    
    if verbose:
        print(f"VIF Analysis Summary:")
        print(f"Total features: {analysis['total_features']}")
        print(f"High multicollinearity (VIF > {vif_threshold}): {analysis['high_multicollinearity_count']}")
        print(f"Moderate multicollinearity (5 < VIF <= {vif_threshold}): {analysis['moderate_multicollinearity_count']}")
        print(f"Max VIF: {analysis['max_vif']:.2f}")
        print(f"Mean VIF: {analysis['mean_vif']:.2f}")
        
        if high_vif:
            print(f"\nHigh VIF features:")
            for feature in high_vif:
                vif_val = vif_df[vif_df["feature"] == feature]["VIF"].iloc[0]
                print(f"  {feature}: {vif_val:.2f}")
        
        print(f"\nMulticollinearity detected: {analysis['multicollinearity_detected']}")
    
    return vif_df, analysis


def get_correlated_features(X: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Find pairs of highly correlated features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    threshold : float, default=0.8
        Correlation threshold
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: feature1, feature2, correlation
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    corr_matrix = X[numeric_cols].corr().abs()
    
    # Get upper triangle pairs
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > threshold:
                pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    return pd.DataFrame(pairs).sort_values('correlation', ascending=False)


def remove_high_vif_features(
    X: pd.DataFrame,
    vif_threshold: float = 10.0,
    max_iterations: int = 100,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list]:
    """
    Iteratively remove the single feature with the highest VIF (> threshold),
    recomputing VIF after each removal, until max VIF <= threshold or we hit
    max_iterations.

    Returns
    -------
    (X_clean, removed_features)
        X_clean : pd.DataFrame with remaining features
        removed_features : list of (feature_name, vif_at_removal)
    """
    X_clean = X.copy()
    removed_features: list[tuple[str, float]] = []

    for it in range(1, max_iterations + 1):
        vif_df, _ = compute_vif(X_clean, vif_threshold=vif_threshold, verbose=False)
        if vif_df.empty or X_clean.shape[1] <= 1:
            if verbose:
                print("No VIF computable (empty or single-feature matrix).")
            break

        # treat inf as very large so they get dropped first
        work = vif_df.copy()
        work["VIF_cmp"] = work["VIF"].replace([np.inf, -np.inf], np.inf)

        # features strictly above threshold
        over = work[work["VIF_cmp"] > vif_threshold]
        if over.empty:
            if verbose:
                max_v = float(work["VIF_cmp"].max())
                print(f"Converged after {it-1} iterations (max VIF = {max_v:.2f} ≤ {vif_threshold}).")
            break

        # drop the worst offender this round
        drop_row = over.sort_values("VIF_cmp", ascending=False).iloc[0]
        feat_to_drop = str(drop_row["feature"])
        vif_to_drop = float(drop_row["VIF"])

        X_clean = X_clean.drop(columns=[feat_to_drop])
        removed_features.append((feat_to_drop, vif_to_drop))

        if verbose:
            print(f"Iteration {it}: removed '{feat_to_drop}' (VIF={vif_to_drop:.2f}); "
                  f"{X_clean.shape[1]} features remain.")

    if verbose:
        final_vif, _ = compute_vif(X_clean, vif_threshold=vif_threshold, verbose=False)
        if not final_vif.empty:
            print(f"Final max VIF: {final_vif['VIF'].max():.2f}")

    return X_clean, removed_features
