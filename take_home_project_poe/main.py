import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import pickle
import asyncio
from pathlib import Path
from datetime import datetime

from src.recorder.poe_trade2_recorder import (
    load_fx_cache_or_raise,
    PriceConverter,
    CURRENCY_CACHE_PATH,
    stream_search_results,
)

from src.research.scripts.pipeline import ModelArtifact
from src.research.scripts.features_poex import make_item_features

OUTPUT_DIR = Path("predictions/undervalued")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# class to log item predictions metadata
class DetectionSession:
    """Track all inspected items with predictions, chosen model, undervalued status."""

    def __init__(self, base_currency: str):
        self.records = []
        self.base_currency = base_currency
        self.logger = logging.getLogger("DetectionSession")

    def log_item(
        self,
        row: dict,
        model_key: str,
        fair_price: float | None,
        current_price: float | None,
        discount: float | None,
        profit: float | None,
        undervalued: bool,
    ):
        record = dict(row)
        record.update({
            "model_key": model_key,
            "fair_price_predicted": fair_price,
            "current_price": current_price,
            "discount_pct": None if discount is None else discount * 100,
            "potential_profit": profit,
            "undervalued": undervalued,
        })
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save(self, path: str | Path):
        df = self.to_dataframe()
        if not df.empty:
            df.to_parquet(path, index=False)
            self.logger.info(f"Session saved to {path}")
        else:
            self.logger.warning("Session empty, nothing to save.")


# class that makes price predictions
class POEArbitrageDetector:

    def __init__(self,
                 base_currency: str = "Exalted Orb",
                 models_path: str = "models/wand"):
        self.base_currency = base_currency
        self.models_path = Path(models_path)
        self.models = {}
        self.fx_cache = None
        self.converter = None
        self.logger = logging.getLogger("POEArbitrageDetector")

        self._load_fx_cache()
        self._load_models()
        self.logger.info(
            f"Initialized detector with base currency: {base_currency}")

    def _load_fx_cache(self):
        self.fx_cache = load_fx_cache_or_raise(CURRENCY_CACHE_PATH)
        self.converter = PriceConverter(self.fx_cache, self.base_currency)
        self.logger.info("FX cache loaded successfully")

    def _load_models(self):
        if not self.models_path.exists():
            raise RuntimeError(f"Models path not found: {self.models_path}")
        for model_file in self.models_path.glob("*.pkl"):
            try:
                with open(model_file, "rb") as f:
                    artifact = pickle.load(f)
                if isinstance(artifact, dict):
                    artifact = ModelArtifact(**artifact)
                if not isinstance(artifact, ModelArtifact):
                    raise TypeError(f"{model_file} is not a ModelArtifact")
                self.models[model_file.stem] = artifact
                self.logger.info(f"Loaded model: {model_file.stem}")
            except Exception as e:
                self.logger.warning(f"Failed to load {model_file}: {e}")
        if not self.models:
            raise RuntimeError("No models loaded successfully")

    # heuristic to determine which wand ML model to apply for a given item
    def _determine_model_key(self, row: dict) -> str:
        rarity = str(row.get("rarity", "")).lower()
        name = str(row.get("name", "") or row.get("type_line", ""))
        if rarity in ["normal", "rare", "magic"]:
            return "craftables_model"
        if rarity == "unique" and name:
            candidate = f"{name.strip()}_model"
            if candidate in self.models:
                return candidate
            for key in self.models:
                if key.endswith("_model") and key != "craftables_model":
                    if key.replace("_model",
                                   "").replace("_",
                                               " ").lower() in name.lower():
                        return key
            return "craftables_model"
        return "craftables_model"

    def _predict_fair_price(self, row: dict) -> tuple[str, float | None]:
        key = self._determine_model_key(row)
        artifact = self.models.get(key)
        if artifact is None:
            return key, None

        if not hasattr(artifact, "model") or not hasattr(
                artifact.model, "predict"):
            self.logger.error(f"Artifact {key} missing model/predict")
            return key, None

        try:
            feats, _, _ = make_item_features(pd.DataFrame([row]), augment=True)
            feats_aligned = artifact.transform_input(feats)
            log_pred = artifact.model.predict(feats_aligned)[0]

            # Predicted price in Exalted Orbs (model’s native scale)
            numerical_price = float(np.expm1(log_pred))

            try:
                # Convert into requested base currency
                fair_price, rate = self.converter.convert(
                    numerical_price, "Exalted Orb")
                self.logger.info(
                    f"Prediction | Item={row.get('name','Unknown')}\n"
                    f"Predicted Log Price={log_pred:.6f},\n"
                    f"Predicted Fair Price in Exalted Orbs={numerical_price:.6f},\n"
                    f"Predicted Fair Price in {self.base_currency}={fair_price:.6f},\n"
                    f"FX Rate=Exalted Orb -> {self.base_currency} ({rate:.6f})\n"
                )
                return key, fair_price
            except Exception as fx_err:
                self.logger.error(
                    f"Prediction FX failed | Item={row.get('name','Unknown')}\n"
                    f"Predicted Log Price={log_pred:.6f}\n"
                    f"Predicted Fair Price in Exalted Orbs={numerical_price:.6f},\n"
                    f"Reason={fx_err}\n")
                return key, None

        except Exception as e:
            self.logger.error(
                f"Prediction failed for {row.get('name', 'Unknown')}: {e}")
            return key, None

    async def stream_detection(
        self,
        payload: dict,
        discount_threshold: float,
        max_results: int,
    ) -> tuple[pd.DataFrame, DetectionSession]:
        session = DetectionSession(self.base_currency)
        undervalued_records = []

        async for rec in stream_search_results(
                payload=payload,
                base_currency=self.base_currency,
                max_results=max_results,
        ):
            row = rec.to_row()
            item_name = row.get("name") or row.get("type_line")
            price_now = row.get("price_amount_in_base")

            if price_now is None:
                session.log_item(row, "N/A", None, price_now, None, None, False)
                continue

            model_key, fair_price = self._predict_fair_price(row)
            if fair_price is None or fair_price <= 0:
                session.log_item(row, model_key, fair_price, price_now, None,
                                 None, False)
                continue

            discount = (fair_price - price_now) / fair_price
            profit = fair_price - price_now
            undervalued = discount >= discount_threshold

            session.log_item(row, model_key, fair_price, price_now, discount,
                             profit, undervalued)

            if undervalued:
                self.logger.info(
                    f"Undervalued | Item={item_name}\n"
                    f"Item Price in {self.base_currency}={price_now:.3f},\n"
                    f"Predicted Fair Price in {self.base_currency}={fair_price:.3f},\n"
                    f"Discount={discount:.2%},\n"
                    f"Profit={profit:.3f} {self.base_currency}\n")
                undervalued_records.append(session.records[-1])
            else:
                self.logger.info(
                    f"Not Undervalued | Item={item_name}\n"
                    f"Item Price in {self.base_currency}={price_now:.3f},\n"
                    f"Predicted Fair Price in {self.base_currency}={fair_price:.3f},\n"
                    f"Discount={discount:.2%},\n"
                    f"Profit={profit:.3f} {self.base_currency}\n")

        df = pd.DataFrame(undervalued_records)
        if not df.empty and "potential_profit" in df.columns:
            df = df.sort_values("potential_profit", ascending=False)
        return df, session


def parse_args():
    parser = argparse.ArgumentParser(description="POE Arbitrage Detection")
    payload_group = parser.add_mutually_exclusive_group()
    payload_group.add_argument("--payload",
                               type=str,
                               help="JSON payload string")
    payload_group.add_argument("--payload-file",
                               type=str,
                               help="File containing JSON payload")

    parser.add_argument("--discount-threshold", type=float, default=0.25)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--base-currency", type=str, default="Exalted Orb")
    parser.add_argument("--models-path", type=str, default="models/wand")
    parser.add_argument("--log-level",
                        type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO")
    return parser.parse_args()


def load_payload(args) -> dict | None:
    if args.payload:
        return json.loads(args.payload)
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    (OUTPUT_DIR / "parq").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "session").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "logs" / f"arbitrage_{ts}.log")
        ],
        force=True)

    logger = logging.getLogger("POEArbMain")
    logger.info("=== Starting POE Arbitrage Detection ===")

    detector = POEArbitrageDetector(base_currency=args.base_currency,
                                    models_path=args.models_path)
    payload = load_payload(args)

    df, session = asyncio.run(
        detector.stream_detection(
            payload=payload,
            discount_threshold=args.discount_threshold,
            max_results=args.max_results,
        ))

    logger.info(f"Undervalued items found: {len(df)}")

    out_parquet = OUTPUT_DIR / "parq" / f"undervalued_{ts}.parquet"
    df.to_parquet(out_parquet, index=False)
    logger.info(f"Undervalued items saved to: {out_parquet}")

    session_path = OUTPUT_DIR / "session" / f"session_{ts}.parquet"
    session.save(session_path)


if __name__ == "__main__":
    main()
