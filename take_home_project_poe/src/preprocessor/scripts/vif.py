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


def remove_high_vif_features(X: pd.DataFrame, vif_threshold: float = 10.0, max_iterations: int = 10):
    """
    Iteratively remove features with highest VIF until all are below threshold.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    vif_threshold : float, default=10.0
        VIF threshold
    max_iterations : int, default=10
        Maximum iterations to prevent infinite loops
    
    Returns:
    --------
    tuple[pd.DataFrame, list]
        - Cleaned feature matrix
        - List of removed features
    """
    X_clean = X.copy()
    removed_features = []
    
    for iteration in range(max_iterations):
        vif_df, analysis = compute_vif(X_clean, vif_threshold=vif_threshold, verbose=False)
        
        if not analysis['multicollinearity_detected']:
            print(f"Converged after {iteration} iterations")
            break
        
        # Remove feature with highest VIF
        highest_vif_feature = vif_df.iloc[0]['feature']
        highest_vif_value = vif_df.iloc[0]['VIF']
        
        X_clean = X_clean.drop(columns=[highest_vif_feature])
        removed_features.append((highest_vif_feature, highest_vif_value))
        
        print(f"Iteration {iteration + 1}: Removed {highest_vif_feature} (VIF: {highest_vif_value:.2f})")
    
    print(f"\nFinal result: {X_clean.shape[1]} features remaining, {len(removed_features)} removed")
    return X_clean, removed_features
