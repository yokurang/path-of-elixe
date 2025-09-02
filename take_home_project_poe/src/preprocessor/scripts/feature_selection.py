import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

def prepare_design_matrix(
    X: pd.DataFrame,
    *,
    one_hot_prefixes: Tuple[str, ...] = ("rarity_",),
    baseline_preferences: Optional[Dict[str, str]] = None,
    drop_zero_variance: bool = True,
    inplace: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepare feature matrix for modeling by handling dummy variable traps and zero variance.
    
    Core functionality:
    1. Drop zero-variance columns (with logging)
    2. Remove one baseline per one-hot encoded family to avoid dummy variable trap
    
    Note: Socket dummies typically already have baseline dropped during feature creation.
    
    Args:
        X: Feature matrix
        one_hot_prefixes: Prefixes that identify one-hot encoded feature families
        baseline_preferences: Preferred baselines to drop per prefix 
                             (e.g., {"rarity_": "rarity_normal"})
        drop_zero_variance: Whether to remove zero-variance features
        inplace: Whether to modify input DataFrame directly
        verbose: Whether to print processing details
        
    Returns:
        (cleaned_features, processing_info)
    """
    # Work on copy unless inplace specified
    features = X if inplace else X.copy()
    
    # Set default baseline preferences
    if baseline_preferences is None:
        baseline_preferences = {
            "rarity_": "rarity_normal"
        }
    
    processing_info = {
        "zero_variance_dropped": [],
        "baselines_dropped": [],
        "already_baseline_dropped": [],
        "warnings": []
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
            continue  # No dummy trap possible with ≤1 column
        
        # Check if this is a complete one-hot encoding (rows sum to 1)
        family_data = features[family_columns]
        row_sums = family_data.sum(axis=1)
        
        # Check if baseline already dropped (row sums < 1 consistently)
        max_row_sum = row_sums.max()
        if max_row_sum < 1.0 and (row_sums == max_row_sum).all():
            # Baseline already dropped - all rows have same sum < 1
            processing_info["already_baseline_dropped"].append(prefix)
            if verbose:
                print(f"Baseline already dropped for '{prefix}' family "
                      f"(max row sum: {max_row_sum:.1f})")
            continue
        
        if (row_sums == 1).all():
            # This is a complete 1-of-K encoding - drop one baseline
            
            # Choose baseline to drop
            preferred_baseline = baseline_preferences.get(prefix)
            if preferred_baseline and preferred_baseline in family_columns:
                baseline_to_drop = preferred_baseline
            else:
                # Fall back to lexicographically first (most predictable)
                baseline_to_drop = sorted(family_columns)[0]
            
            features.drop(columns=[baseline_to_drop], inplace=True)
            processing_info["baselines_dropped"].append((prefix, baseline_to_drop))
            
            if verbose:
                remaining = len(family_columns) - 1
                print(f"Dropped baseline '{baseline_to_drop}' from {prefix} family "
                      f"({remaining} features remaining)")
        
        elif row_sums.nunique() > 1 and verbose:
            # Partial encoding - warn but don't modify
            warning = f"Prefix '{prefix}' appears to be partial one-hot encoding (varying row sums)"
            processing_info["warnings"].append(warning)
            print(f"Warning: {warning}")
    
    # Step 3: Final validation
    remaining_features = len(features.columns)
    if verbose:
        print(f"\nFinal feature count: {remaining_features}")
    
    return features, processing_info

def validate_design_matrix(
    X: pd.DataFrame,
    y: Optional[pd.DataFrame] = None,
    *,
    check_multicollinearity: bool = False,
    correlation_threshold: float = 0.99
) -> Dict[str, Any]:
    """
    Validate design matrix for common modeling issues.
    
    Args:
        X: Feature matrix
        y: Optional target variables  
        check_multicollinearity: Whether to check for highly correlated features
        correlation_threshold: Correlation threshold for multicollinearity detection
        
    Returns:
        Validation results dictionary
    """
    validation = {
        "shape": X.shape,
        "has_missing": False,
        "has_infinite": False,
        "zero_variance_features": [],
        "constant_features": [],
        "highly_correlated_pairs": []
    }
    
    # Check for missing values
    if X.isna().any().any():
        validation["has_missing"] = True
        missing_cols = X.columns[X.isna().any()].tolist()
        validation["missing_columns"] = missing_cols
        print(f"Warning: {len(missing_cols)} columns have missing values")
    
    # Check for infinite values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_mask = np.isinf(X[numeric_cols]).any()
        if inf_mask.any():
            validation["has_infinite"] = True
            inf_cols = numeric_cols[inf_mask].tolist()
            validation["infinite_columns"] = inf_cols
            print(f"Warning: {len(inf_cols)} columns have infinite values")
    
    # Check for zero variance (shouldn't happen after prepare_design_matrix)
    if len(numeric_cols) > 0:
        variances = X[numeric_cols].var(ddof=0)
        zero_var = variances[variances == 0].index.tolist()
        validation["zero_variance_features"] = zero_var
        
        if zero_var:
            print(f"Warning: {len(zero_var)} features still have zero variance")
    
    # Optional multicollinearity check
    if check_multicollinearity and len(numeric_cols) > 1:
        corr_matrix = X[numeric_cols].corr().abs()
        
        # Find pairs with correlation above threshold
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlation = corr_matrix.iloc[i, j]
                if correlation >= correlation_threshold:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, correlation))
        
        validation["highly_correlated_pairs"] = high_corr_pairs
        if high_corr_pairs and len(high_corr_pairs) <= 10:
            print(f"\nHigh correlations (>{correlation_threshold}):")
            for col1, col2, corr in high_corr_pairs:
                print(f"  {col1} ↔ {col2}: {corr:.3f}")
        elif len(high_corr_pairs) > 10:
            print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs")
    
    # Shape validation with targets
    if y is not None and X.shape[0] != y.shape[0]:
        validation["shape_mismatch"] = True
        print(f"Warning: Shape mismatch - X: {X.shape[0]} rows, y: {y.shape[0]} rows")
    
    return validation

def get_feature_summary(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Get concise summary of feature matrix characteristics.
    
    Returns:
        Dictionary with feature type counts and key statistics
    """
    summary = {"shape": X.shape}
    
    # Feature type breakdown
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
    
    # Basic statistics for numeric features
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
    """Print a concise, readable summary of the feature matrix."""
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