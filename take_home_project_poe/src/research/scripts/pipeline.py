import os
import sys
import pickle
import random
import numpy as np
import logging
import pandas as pd
from pathlib import Path

from .utils import combine_parquet_files, summarize_feature_set
from .features_poex import prepare_feature_sets_from_raw, validate_features
from .features_preperation import prepare_design_matrix, validate_design_matrix, print_feature_summary
from .vif import remove_high_vif_features
from .modelling import compare_regularized_models


# -------------------------------
# Helper class to wrap models
# -------------------------------
class ModelArtifact:
    def __init__(self, model, features, vif_info=None, metadata=None):
        self.model = model                 # sklearn Pipeline (scaler inside)
        self.features = features           # aligned feature names
        self.vif_info = vif_info           # diagnostic info
        self.metadata = metadata           # optional metadata frame

    def transform_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align input DataFrame to training features. Scaling is handled inside the pipeline."""
        return X.reindex(columns=self.features, fill_value=0)

    def predict(self, X: pd.DataFrame):
        """Predict log_price from raw feature DataFrame."""
        X_prepared = self.transform_input(X)
        return self.model.predict(X_prepared)


# -------------------------------
# Config
# -------------------------------
random.seed(42)
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------------
# Craftables pipeline
# -------------------------------
def run_pipeline_craftables(file_paths: list[str], output_file: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory {output_dir} created or already exists.")

    try:
        combine_parquet_files(file_paths=file_paths, output_file=output_file)
        logging.info(f"Files combined and saved to {output_file}")
    except Exception as e:
        logging.error(f"Error combining parquet files: {str(e)}")
        return

    try:
        res = prepare_feature_sets_from_raw(output_file, group_uniques_by_name=False)
        if "craft_all" not in res:
            logging.error(f"Missing expected 'craft_all' in result keys {res.keys()}")
            return
        Xc, yc, mc = res["craft_all"]
    except Exception as e:
        logging.error(f"Error preparing feature sets: {str(e)}")
        return

    try:
        validate_features(Xc, yc)
        summarize_feature_set(Xc, yc, mc, show_variance_analysis=True)
    except Exception as e:
        logging.error(f"Feature validation error: {str(e)}")
        return

    try:
        Xc_cleaned, info_craftable = prepare_design_matrix(
            Xc, one_hot_prefixes=(), drop_zero_variance=True,
            standardize=True, standardize_fit_stats=None, verbose=True
        )
    except Exception as e:
        logging.error(f"Error preparing design matrix: {str(e)}")
        return

    try:
        Xc_cleaned, _ = remove_high_vif_features(Xc_cleaned)
    except Exception as e:
        logging.error(f"Error in VIF filtering: {str(e)}")
        return

    try:
        validate_design_matrix(Xc_cleaned, yc, check_multicollinearity=True)
        print_feature_summary(Xc_cleaned)
    except Exception as e:
        logging.error(f"Error validating design matrix: {str(e)}")
        return

    try:
        cmp_craftables = compare_regularized_models(
            Xc_cleaned, yc["log_price"], metadata=mc,
            ridge_alphas=np.logspace(-6, 1, 18),
            en_alphas=np.logspace(-6, -2, 10),
            en_l1_ratios=[0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95],
            cv_splits=5, test_size=0.2,
            random_state=42, n_jobs=-1, epsilon=1.35, plots=False
        )

        # ✅ Pick the model with the best test R²
        ridge_r2 = cmp_craftables["Ridge"]["metrics"]["r2_test"]
        enet_r2 = cmp_craftables["ElasticNet"]["metrics"]["r2_test"]

        if ridge_r2 >= enet_r2:
            best_model = cmp_craftables["Ridge"]["best_model"]
            logging.info(f"Chose Ridge model (test R²={ridge_r2:.4f} ≥ {enet_r2:.4f})")
        else:
            best_model = cmp_craftables["ElasticNet"]["best_model"]
            logging.info(f"Chose ElasticNet model (test R²={enet_r2:.4f} > {ridge_r2:.4f})")

        artifact = ModelArtifact(
            model=best_model,
            features=list(Xc_cleaned.columns),
            vif_info=info_craftable,
            metadata=mc,
        )
        with open(output_dir / "craftables_model.pkl", "wb") as f:
            pickle.dump(artifact, f)
        logging.info(f"Craftables model trained and saved to {output_dir/'craftables_model.pkl'}")
    except Exception as e:
        logging.error(f"Error training craftables model: {str(e)}")
        return


# -------------------------------
# Uniques pipeline
# -------------------------------
def run_pipeline_uniques(file_paths: list[str], output_file: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory {output_dir} created or already exists.")

    try:
        combine_parquet_files(file_paths=file_paths, output_file=output_file)
    except Exception as e:
        logging.error(f"Error combining parquet files: {str(e)}")
        return

    try:
        res = prepare_feature_sets_from_raw(output_file, group_uniques_by_name=True)
        if "unique_by_name" not in res:
            logging.error(f"Missing 'unique_by_name' in result keys {res.keys()}")
            return
        unique_item_features = res["unique_by_name"]
    except Exception as e:
        logging.error(f"Error preparing feature sets: {str(e)}")
        return

    for name, data in unique_item_features.items():
        logging.info(f"Processing unique item: {name}")
        X_unique, y_unique, unique_meta = data

        try:
            X_unique_cleaned, info_unique = prepare_design_matrix(
                X_unique, one_hot_prefixes=(), drop_zero_variance=True,
                standardize=True, standardize_fit_stats=None, verbose=True
            )
        except Exception as e:
            logging.error(f"Error preparing design matrix for {name}: {str(e)}")
            continue

        try:
            X_unique_cleaned, _ = remove_high_vif_features(X_unique_cleaned)
        except Exception as e:
            logging.error(f"Error in VIF filtering for {name}: {str(e)}")
            continue

        try:
            validate_design_matrix(X_unique_cleaned, y_unique, check_multicollinearity=True)
            print_feature_summary(X_unique_cleaned)
        except Exception as e:
            logging.error(f"Validation error for {name}: {str(e)}")
            continue

        try:
            cmp_uniques = compare_regularized_models(
                X_unique_cleaned, y_unique["log_price"], metadata=unique_meta,
                ridge_alphas=np.logspace(-6, 1, 18),
                en_alphas=np.logspace(-6, -2, 10),
                en_l1_ratios=[0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95],
                cv_splits=5, test_size=0.2,
                random_state=42, n_jobs=-1, epsilon=1.35, plots=False
            )

            ridge_r2 = cmp_uniques["Ridge"]["metrics"]["r2_test"]
            enet_r2 = cmp_uniques["ElasticNet"]["metrics"]["r2_test"]

            if ridge_r2 >= enet_r2:
                best_model = cmp_uniques["Ridge"]["best_model"]
                logging.info(f"Chose Ridge for {name} (test R²={ridge_r2:.4f} ≥ {enet_r2:.4f})")
            else:
                best_model = cmp_uniques["ElasticNet"]["best_model"]
                logging.info(f"Chose ElasticNet for {name} (test R²={enet_r2:.4f} > {ridge_r2:.4f})")

            artifact = ModelArtifact(
                model=best_model,
                features=list(X_unique_cleaned.columns),
                vif_info=info_unique,
                metadata=unique_meta,
            )
            model_filename = output_dir / f"{name}_model.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(artifact, f)
            logging.info(f"Unique model for {name} saved to {model_filename}")
        except Exception as e:
            logging.error(f"Error training or saving model for {name}: {str(e)}")
            continue


if __name__ == "__main__":
    file_paths = [
        "/Users/yokurang/Documents/Recruiting/AlphaGrep/path-of-elix/take_home_project_poe/training_data/overall/weapon_wand_overall_rise_20of_20the_20abyssal.parquet",
        "/Users/yokurang/Documents/Recruiting/AlphaGrep/path-of-elix/take_home_project_poe/training_data/overall/weapon_wand_overall_standard.parquet",
    ]
    output_file = "/Users/yokurang/Documents/Recruiting/AlphaGrep/path-of-elix/take_home_project_poe/training_data/overall/total/wand.parquet"
    output_dir = Path("models/wand/")

    run_pipeline_craftables(file_paths, output_file, output_dir)
    run_pipeline_uniques(file_paths, output_file, output_dir)
