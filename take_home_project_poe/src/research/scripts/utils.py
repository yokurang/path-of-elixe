from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

def setup_logging(level: str = "INFO") -> str:
    """
    Initialize logging to console + logs/log_<YYYY-MM-DD-HH-MM>.txt.
    Returns the path to the log file.
    """
    os.makedirs("logs", exist_ok=True)
    log_path = datetime.now().strftime("logs/log_%Y-%m-%d-%H-%M.txt")

    # If logging already configured, don't reconfigure; just add a file handler.
    root = logging.getLogger()
    has_handlers = bool(root.handlers)

    lvl = getattr(logging, level.upper(), logging.INFO)
    if not has_handlers:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, encoding="utf-8")
            ],
        )
    else:
        # Ensure our file handler exists
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(lvl)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.setLevel(lvl)

    logging.info("Logging initialized; file=%s", log_path)
    return log_path

# -----------------------------------------------------------------------------
# Path Resolution
# -----------------------------------------------------------------------------
def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find project root by walking up directory tree.
    
    Looks for:
    1. POE_PROJECT_ROOT environment variable
    2. Directory containing 'training_data' folder
    3. Directory named 'take_home_project_poe' 
    4. Git repository root (.git folder)
    """
    # Environment variable override
    if env_root := os.environ.get("POE_PROJECT_ROOT"):
        return Path(env_root).expanduser().resolve()
    
    # Walk up directory tree
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        indicators = [
            path / "training_data",
            path / ".git"
        ]
        if any(indicator.exists() for indicator in indicators) or path.name == "take_home_project_poe":
            return path
    
    return current

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_parquet_robust(file_path: Path) -> pd.DataFrame:
    """Load parquet file with engine fallback for maximum compatibility."""
    engines = ["pyarrow", "fastparquet"]
    
    for engine in engines:
        try:
            return pd.read_parquet(file_path, engine=engine)
        except Exception:
            continue
    
    raise RuntimeError(
        f"Failed to read {file_path} with available engines ({engines}). "
        "Install pyarrow or fastparquet."
    )

def preview_dataset(
    file_path: str | Path,
    *,
    base_path: Optional[str | Path] = None,
    n_rows: int = 5,
    show_dtypes: bool = False
) -> pd.DataFrame:
    """
    Load and preview a parquet dataset with clean, structured output.
    
    Args:
        file_path: Path to parquet file (relative to base_path if not absolute)
        base_path: Base directory for relative paths
        n_rows: Number of rows to show in preview
        show_dtypes: Whether to display column data types
    """
    # Resolve file path
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        root = Path(base_path).expanduser().resolve() if base_path else find_project_root()
        path = (root / path).resolve()
    
    # Handle directory input (find single parquet file)
    if path.is_dir():
        parquet_files = list(path.glob("*.parquet"))
        if len(parquet_files) != 1:
            raise FileNotFoundError(
                f"Directory {path} must contain exactly one .parquet file "
                f"(found {len(parquet_files)})"
            )
        path = parquet_files[0]
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Load data
    df = load_parquet_robust(path)
    
    # Print summary
    print(f"Dataset: {path.name}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    if show_dtypes:
        print(f"\nColumn Types:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
    
    print(f"\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")
    
    # Show preview with controlled formatting
    print(f"\nFirst {n_rows} rows:")
    with pd.option_context(
        "display.max_columns", None,
        "display.max_colwidth", 50,  # Limit column width for readability
        "display.width", None,
        "display.expand_frame_repr", False,
    ):
        print(df.head(n_rows).to_string(index=False))
    
    return df

# -----------------------------------------------------------------------------
# Feature Analysis
# -----------------------------------------------------------------------------
def analyze_features(
    features: pd.DataFrame, 
    targets: pd.DataFrame, 
    metadata: Optional[pd.DataFrame] = None,
    *,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze processed features for data quality and characteristics.
    
    Args:
        features: Feature matrix
        targets: Target variables
        metadata: Optional metadata DataFrame
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary containing analysis results
    """
    analysis = {}
    
    # Basic shape info
    analysis["shapes"] = {
        "features": features.shape,
        "targets": targets.shape,
        "metadata": metadata.shape if metadata is not None else None
    }
    
    # Data quality checks
    def _count_issues(df: pd.DataFrame) -> Tuple[int, int]:
        values = df.to_numpy()
        nan_count = int(pd.isna(values).sum())
        inf_count = int(np.isinf(values).sum()) if np.issubdtype(values.dtype, np.number) else 0
        return nan_count, inf_count
    
    features_nan, features_inf = _count_issues(features)
    targets_nan, targets_inf = _count_issues(targets)
    
    analysis["data_quality"] = {
        "features_nan": features_nan,
        "features_inf": features_inf, 
        "targets_nan": targets_nan,
        "targets_inf": targets_inf
    }
    
    # Feature type analysis
    numeric_features = [col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])]
    
    if numeric_features:
        feature_stats = features[numeric_features].std(ddof=0)
        zero_variance = feature_stats[feature_stats == 0].index.tolist()
        low_variance = feature_stats[(feature_stats > 0) & (feature_stats < 0.01)].index.tolist()
        
        analysis["feature_variance"] = {
            "zero_variance_count": len(zero_variance),
            "low_variance_count": len(low_variance),
            "zero_variance_features": zero_variance,
            "low_variance_features": low_variance
        }
    
    # One-hot encoding analysis
    rarity_features = [col for col in features.columns if col.startswith("rarity_")]
    category_features = [col for col in features.columns if col.startswith("cat_")]
    socket_features = [col for col in features.columns if col.startswith("sockets_")]
    
    analysis["encodings"] = {
        "rarity_features": rarity_features,
        "category_features": category_features,
        "socket_features": socket_features
    }
    
    # Target analysis
    if not targets.empty:
        target_stats = {}
        for col in targets.columns:
            if pd.api.types.is_numeric_dtype(targets[col]):
                target_stats[col] = {
                    "mean": float(targets[col].mean()),
                    "std": float(targets[col].std()),
                    "min": float(targets[col].min()),
                    "max": float(targets[col].max()),
                    "median": float(targets[col].median())
                }
        analysis["target_stats"] = target_stats
    
    # Print summary if verbose
    if verbose:
        _print_analysis_summary(analysis)
    
    return analysis

def _print_analysis_summary(analysis: Dict[str, Any]) -> None:
    """Print a clean summary of feature analysis."""
    shapes = analysis["shapes"]
    quality = analysis["data_quality"]
    
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    # Shapes
    print(f"Features: {shapes['features'][0]:,} rows × {shapes['features'][1]} columns")
    print(f"Targets:  {shapes['targets'][0]:,} rows × {shapes['targets'][1]} columns")
    if shapes.get("metadata"):
        print(f"Metadata: {shapes['metadata'][0]:,} rows × {shapes['metadata'][1]} columns")
    
    # Data quality
    print(f"\nData Quality:")
    if quality["features_nan"] > 0 or quality["targets_nan"] > 0:
        print(f"  NaN values - Features: {quality['features_nan']:,}, Targets: {quality['targets_nan']:,}")
    else:
        print(f"  No NaN values detected")
    
    if quality["features_inf"] > 0 or quality["targets_inf"] > 0:
        print(f"  Infinite values - Features: {quality['features_inf']:,}, Targets: {quality['targets_inf']:,}")
    else:
        print(f"  No infinite values detected")
    
    # Feature variance
    if "feature_variance" in analysis:
        variance = analysis["feature_variance"]
        print(f"\nFeature Variance:")
        print(f"  Zero variance: {variance['zero_variance_count']} features")
        print(f"  Low variance (<0.01): {variance['low_variance_count']} features")
        
        if variance["zero_variance_features"]:
            print(f"  Zero variance features: {', '.join(variance['zero_variance_features'][:5])}")
            if len(variance["zero_variance_features"]) > 5:
                print(f"    ... and {len(variance['zero_variance_features']) - 5} more")
    
    # Encodings
    encodings = analysis["encodings"]
    print(f"\nOne-Hot Encodings:")
    print(f"  Rarity: {len(encodings['rarity_features'])} features")
    print(f"  Sockets: {len(encodings['socket_features'])} features")
    if encodings["category_features"]:
        print(f"  Category: {len(encodings['category_features'])} features (should be empty)")
    
    # Target summary
    if "target_stats" in analysis:
        print(f"\nTarget Statistics:")
        for target, stats in analysis["target_stats"].items():
            print(f"  {target}: μ={stats['mean']:.2f}, σ={stats['std']:.2f}, "
                  f"range=[{stats['min']:.2f}, {stats['max']:.2f}]")

def quick_validation(
    features: pd.DataFrame, 
    targets: pd.DataFrame,
    *,
    check_inf: bool = True,
    check_nan: bool = True
) -> bool:
    """
    Quick validation check for processed data.
    
    Returns True if data passes all checks, False otherwise.
    Prints issues found.
    """
    issues = []
    
    # Shape compatibility
    if features.shape[0] != targets.shape[0]:
        issues.append(f"Shape mismatch: features={features.shape[0]} vs targets={targets.shape[0]} rows")
    
    # NaN checks
    if check_nan:
        features_nan = features.isna().sum().sum()
        targets_nan = targets.isna().sum().sum()
        if features_nan > 0:
            issues.append(f"Features contain {features_nan:,} NaN values")
        if targets_nan > 0:
            issues.append(f"Targets contain {targets_nan:,} NaN values")
    
    # Infinite value checks
    if check_inf:
        features_inf = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        targets_inf = np.isinf(targets.select_dtypes(include=[np.number])).sum().sum()
        if features_inf > 0:
            issues.append(f"Features contain {features_inf:,} infinite values")
        if targets_inf > 0:
            issues.append(f"Targets contain {targets_inf:,} infinite values")
    
    # Print results
    if issues:
        print("Validation Issues:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("Data validation passed")
        return True

# -----------------------------------------------------------------------------
# Streamlined Data Loading and Analysis
# -----------------------------------------------------------------------------
def load_and_analyze(
    file_path: str | Path,
    *,
    base_path: Optional[str | Path] = None,
    preview_rows: int = 3,
    analyze_features: bool = False
) -> pd.DataFrame:
    """
    Load dataset and provide focused analysis output.
    
    Args:
        file_path: Path to parquet file
        base_path: Base directory for relative paths  
        preview_rows: Number of rows to show in preview
        analyze_features: Whether to run detailed feature analysis
        
    Returns:
        Loaded DataFrame
    """
    # Resolve and load
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        root = Path(base_path).expanduser().resolve() if base_path else find_project_root()
        path = (root / path).resolve()
    
    if path.is_dir():
        parquet_files = list(path.glob("*.parquet"))
        if len(parquet_files) != 1:
            raise FileNotFoundError(f"Directory must contain exactly one .parquet file")
        path = parquet_files[0]
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    df = load_parquet_robust(path)
    
    # Concise summary
    print(f"Loaded: {path.name}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Show key columns only
    key_columns = [
        "id", "rarity", "category", "base_type", "ilvl", 
        "price_amount_in_base", "properties", "requirements"
    ]
    available_key_cols = [col for col in key_columns if col in df.columns]
    
    if available_key_cols and preview_rows > 0:
        print(f"\nPreview ({preview_rows} rows, key columns):")
        preview_df = df[available_key_cols].head(preview_rows)
        
        # Truncate long text fields for readability
        for col in preview_df.columns:
            if preview_df[col].dtype == object:
                preview_df[col] = preview_df[col].astype(str).str[:50] + "..."
        
        print(preview_df.to_string(index=False, max_colwidth=50))
    
    if analyze_features:
        # Basic feature type breakdown
        dtypes = df.dtypes.value_counts()
        print(f"\nColumn Types:")
        for dtype, count in dtypes.items():
            print(f"  {dtype}: {count}")
    
    return df

# -----------------------------------------------------------------------------
# Feature Set Reporting  
# -----------------------------------------------------------------------------
def summarize_feature_set(
    features: pd.DataFrame,
    targets: pd.DataFrame, 
    metadata: Optional[pd.DataFrame] = None,
    *,
    show_variance_analysis: bool = False,
    show_sample: bool = True
) -> Dict[str, Any]:
    """
    Generate concise summary of processed feature set.
    
    Args:
        features: Processed feature matrix
        targets: Target variables
        metadata: Optional metadata
        show_variance_analysis: Include detailed variance analysis
        show_sample: Show sample of features and targets
        
    Returns:
        Summary statistics dictionary
    """
    summary = {}
    
    # Basic info
    print("=" * 50)
    print("FEATURE SET SUMMARY")
    print("=" * 50)
    
    print(f"Features: {features.shape[0]:,} samples × {features.shape[1]} features")
    print(f"Targets:  {targets.shape[1]} target variable(s)")
    
    # Data quality check
    is_clean = quick_validation(features, targets, check_inf=True, check_nan=True)
    summary["is_clean"] = is_clean
    
    # Feature type breakdown
    feature_types = _categorize_features(features)
    summary["feature_types"] = feature_types
    
    print(f"\nFeature Categories:")
    for category, feature_list in feature_types.items():
        if feature_list:
            print(f"  {category}: {len(feature_list)}")
    
    # Target analysis
    if not targets.empty:
        print(f"\nTarget Analysis:")
        for col in targets.columns:
            if pd.api.types.is_numeric_dtype(targets[col]):
                stats = targets[col].describe()
                print(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                      f"range=[{stats['min']:.1f}, {stats['max']:.1f}]")
    
    # Optional variance analysis
    if show_variance_analysis:
        _print_variance_analysis(features)
    
    # Optional sample display
    if show_sample and not features.empty:
        print(f"\nSample Features (first 3 rows, first 8 columns):")
        sample_cols = features.columns[:8]
        sample_data = features[sample_cols].head(3)
        print(sample_data.round(3).to_string())
    
    return summary

def _categorize_features(features: pd.DataFrame) -> Dict[str, list]:
    """Categorize features by their naming patterns and types."""
    categories = {
        "dps": [],
        "rarity": [],
        "sockets": [],
        "requirements": [],
        "boolean_flags": [],
        "continuous": [],
        "other": []
    }
    
    for col in features.columns:
        col_str = str(col)
        
        if col_str.endswith("dps") or col_str in ["pdps", "fdps", "cdps", "ldps", "chaos_dps"]:
            categories["dps"].append(col)
        elif col_str.startswith("rarity_"):
            categories["rarity"].append(col)
        elif col_str.startswith("sockets_"):
            categories["sockets"].append(col)
        elif col_str.startswith("req_"):
            categories["requirements"].append(col)
        elif features[col].dtype == bool or set(features[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            categories["boolean_flags"].append(col)
        elif pd.api.types.is_numeric_dtype(features[col]):
            categories["continuous"].append(col)
        else:
            categories["other"].append(col)
    
    return categories

def _print_variance_analysis(features: pd.DataFrame) -> None:
    """Print analysis of feature variance for identifying potential issues."""
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("\nNo numeric features for variance analysis")
        return
    
    variances = features[numeric_cols].var(ddof=0)
    
    zero_var = variances[variances == 0].index.tolist()
    very_low_var = variances[(variances > 0) & (variances < 1e-6)].index.tolist()
    
    print(f"\nVariance Analysis:")
    print(f"  Zero variance: {len(zero_var)} features")
    print(f"  Very low variance: {len(very_low_var)} features")
    
    if zero_var:
        print(f"  Zero variance features: {', '.join(zero_var[:3])}")
        if len(zero_var) > 3:
            print(f"    ... and {len(zero_var) - 3} more")
    
    if len(variances) > 0:
        print(f"  Variance range: [{variances.min():.2e}, {variances.max():.2e}]")

# -----------------------------------------------------------------------------
# Backwards Compatibility (Deprecated)
# -----------------------------------------------------------------------------
def preview_parquet(*args, **kwargs) -> pd.DataFrame:
    """Deprecated: Use preview_dataset instead."""
    print("Warning: preview_parquet is deprecated, use preview_dataset instead")
    return preview_dataset(*args, **kwargs)

def report_dataset(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    meta: pd.DataFrame,
    **kwargs
) -> Dict[str, Any]:
    """Deprecated: Use summarize_feature_set instead."""
    print("Warning: report_dataset is deprecated, use summarize_feature_set instead")
    return summarize_feature_set(X, y, meta, **kwargs)

def combine_parquet_files(file_paths: list[str | Path], output_file: str | Path) -> None:
    """
    Combine multiple Parquet files into one DataFrame and save it as a Parquet file.
    
    Args:
        file_paths: List of file paths to Parquet files
        output_file: Output file path where the combined Parquet file will be saved
    """
    # Resolve paths using find_project_root
    project_root = find_project_root()

    # Load the Parquet files into DataFrames
    dataframes = []
    for file_path in file_paths:
        # Resolve file path using find_project_root logic
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            root = project_root  # Base path is the project root
            path = (root / path).resolve()
        
        # Handle directory input (find single parquet file)
        if path.is_dir():
            parquet_files = list(path.glob("*.parquet"))
            if len(parquet_files) != 1:
                raise FileNotFoundError(
                    f"Directory {path} must contain exactly one .parquet file "
                    f"(found {len(parquet_files)})"
                )
            path = parquet_files[0]
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Print resolved path for debugging
        print(f"Loading file: {path}")
        
        # Load the Parquet file
        df = load_parquet_robust(path)
        dataframes.append(df)
    
    # Concatenate the DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Ensure the output directory exists
    output_path = Path(output_file).expanduser().resolve()
    output_directory = output_path.parent
    if not output_directory.exists():
        os.makedirs(output_directory, exist_ok=True)
    
    # Save the combined DataFrame to the specified file
    combined_df.to_parquet(output_path, engine="pyarrow")
    print(f"Combined DataFrame saved to: {output_path}")