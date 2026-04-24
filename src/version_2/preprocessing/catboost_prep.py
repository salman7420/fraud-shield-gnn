# src/common/model_prep/catboost_prep.py
#
# Stage 2B — CatBoost Data Preparation
# Transforms enriched splits into CatBoost-ready format.
#
# What this file does:
#   1. Loads enriched splits (data/enriched/)
#   2. Identifies categorical columns (object dtype)
#   3. Fills categorical NaN → "missing" string
#      (numeric NaN left untouched — CatBoost handles natively)
#   4. Casts binary flag columns to int
#   5. Saves cat_features list → artifacts/catboost_cat_features.json
#   6. Saves CatBoost-ready splits → data/versions/v2_catboost/
#
# Usage (from project ROOT):
#   python -m src.common.model_prep.catboost_prep

import sys
import json
import time
import traceback
from pathlib import Path

from src.data_ingestion.load_data import load_data, save_data
import numpy as np
import pandas as pd

# ── Project root on path ───────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.data_configs import (
    ENRICHED,
    ARTIFACTS,
    V2_TRAIN,
    V2_VAL,
    V2_TEST,
)

logger = get_logger(__name__)

# ── Paths ──────────────────────────────────────────────────
ENRICHED_DIR  = ENRICHED
ARTIFACTS_DIR = ARTIFACTS
CAT_FEATURES_PATH = ARTIFACTS_DIR / "catboost_cat_features.json"

# Output paths from data_configs
V2_DIR = V2_TRAIN.parent   # data/versions/v2_catboost/

SPLITS = ["train", "val", "test"]

# ── Null fill value for categorical columns ────────────────
# CatBoost requires string values for cat cols — NaN causes error
# "missing" becomes its own category and CatBoost learns its signal
CAT_NULL_FILL = "missing"


# ══════════════════════════════════════════════════════════
#  STEP 1 — LOAD ENRICHED SPLITS
# ══════════════════════════════════════════════════════════

def load_enriched_splits() -> dict[str, pd.DataFrame]:
    """
    Loads enriched train/val/test from data/enriched/.
    Validates each file exists and is non-empty.
    """
    splits = {}
    for name in SPLITS:
        path = ENRICHED_DIR / f"{name}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Enriched split not found: {path}\n"
                f"Run build_enriched.py first."
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Enriched split is empty: {path}")

        logger.info(f"Loading enriched {name}.csv ...")
        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"Loaded dataframe is empty: {path}")

        logger.info(
            f"  {name}: {df.shape[0]:,} rows × {df.shape[1]:,} cols"
        )
        splits[name] = df

    return splits



# ══════════════════════════════════════════════════════════
#  STEP 2 — IDENTIFY CATEGORICAL COLUMNS
# ══════════════════════════════════════════════════════════

def identify_cat_features(df: pd.DataFrame) -> list[str]:
    """
    Detects categorical columns by dtype (object or string).
    Excludes isFraud (target column).
    Returns sorted list of categorical column names.
    """
    cat_features = sorted([
        col for col in df.select_dtypes(include=["object", "string"]).columns
        if col != "isFraud"
    ])

    logger.info(
        f"Categorical features identified: {len(cat_features)} columns\n"
        f"  {cat_features}"
    )
    return cat_features


# ══════════════════════════════════════════════════════════
#  STEP 3 — FILL CATEGORICAL NaN
# ══════════════════════════════════════════════════════════

def fill_categorical_nulls(
    df: pd.DataFrame,
    cat_features: list[str],
    split_name: str,
) -> pd.DataFrame:
    """
    Fills NaN in categorical columns with CAT_NULL_FILL ("missing").

    Why only categorical cols:
      - Numeric NaN → CatBoost handles natively (no action needed)
      - Categorical NaN → CatBoost raises error (string expected)

    Logs null counts before and after for transparency.

    Edge cases:
      - cat_feature not in df → skips with warning
      - Column already fully non-null → logs 0 filled, no change
    """
    for col in cat_features:
        if col not in df.columns:
            logger.warning(
                f"[{split_name}] Cat feature '{col}' not in dataframe — skipping"
            )
            continue

        null_count = df[col].isna().sum()
        if null_count > 0:
            df[col] = df[col].fillna(CAT_NULL_FILL).astype(str)
            pct = null_count / len(df) * 100
            logger.info(
                f"[{split_name}] '{col}': filled {null_count:,} "
                f"NaN ({pct:.1f}%) → '{CAT_NULL_FILL}'"
            )
        else:
            # Still cast to str to ensure consistent dtype
            df[col] = df[col].astype(str)

    return df


# ══════════════════════════════════════════════════════════
#  STEP 4 — CAST BINARY FLAG COLUMNS TO INT
# ══════════════════════════════════════════════════════════

def cast_binary_flags(
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Ensures all binary engineered flag columns are int dtype.

    Why: Some flag columns may be float (0.0/1.0/NaN) after
    pandas read_csv or FE operations. CatBoost works with both
    but int is cleaner and slightly more memory efficient.

    Flag columns that may need casting:
      is_weekend, is_high_risk_hour, id_data_present,
      id_13_was_null, id_16_was_null, card1_is_high_freq,
      amt_is_very_low, amt_is_high, card_device_is_high

    Note: addr_is_unique and device_txn_count can be NaN
    → These are NOT cast (would convert NaN to int error)
    → Left as float — CatBoost handles numeric NaN natively
    """
    # Only cast cols that should be fully non-null binary flags
    BINARY_FLAG_COLS = [
        "is_weekend",
        "is_high_risk_hour",
        "id_data_present",
        "id_13_was_null",
        "id_16_was_null",
        "card1_is_high_freq",
        "amt_is_very_low",
        "amt_is_high",
        "card_device_is_high",
    ]

    cast_count = 0
    for col in BINARY_FLAG_COLS:
        if col not in df.columns:
            continue  # silently skip — may not be in this split

        if df[col].isna().any():
            logger.warning(
                f"[{split_name}] Binary flag '{col}' has "
                f"{df[col].isna().sum():,} NaN — skipping int cast"
            )
            continue

        if df[col].dtype != "int64":
            df[col] = df[col].astype(int)
            cast_count += 1

    if cast_count > 0:
        logger.info(
            f"[{split_name}] Cast {cast_count} binary flag cols to int"
        )
    return df


# ══════════════════════════════════════════════════════════
#  STEP 5 — VALIDATE CATBOOST READINESS
# ══════════════════════════════════════════════════════════

def validate_catboost_ready(
    splits: dict[str, pd.DataFrame],
    cat_features: list[str],
) -> None:
    """
    Final checks before saving:
      1. No NaN remaining in categorical columns
      2. Categorical columns are string dtype
      3. Numeric columns have correct dtypes (float/int)
      4. isFraud present in train/val, absent in test
      5. Log full null report for numeric cols (informational)
    """
    logger.info("Validating CatBoost readiness ...")
    has_errors = False

    for name, df in splits.items():

        # Check 1 & 2: categorical columns — no NaN, string dtype
        for col in cat_features:
            if col not in df.columns:
                continue
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.error(
                    f"[{name}] Categorical col '{col}' still has "
                    f"{null_count:,} NaN after fill — CatBoost will error"
                )
                has_errors = True
            if df[col].dtype not in ["object", "string"]:
                logger.error(
                    f"[{name}] Categorical col '{col}' dtype is "
                    f"'{df[col].dtype}' — expected string/object"
                )
                has_errors = True

        # Check 3: isFraud presence
        if name in ["train", "val"] and "isFraud" not in df.columns:
            logger.error(f"[{name}] isFraud missing — cannot train")
            has_errors = True
        if name == "test" and "isFraud" in df.columns:
            logger.error(
                "[test] isFraud found in test split — "
                "should have been removed in build_enriched.py"
            )
            has_errors = True

        # Check 4: numeric null report (informational only)
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns
        num_nulls = {
            c: int(df[c].isna().sum())
            for c in num_cols
            if df[c].isna().sum() > 0 and c != "isFraud"
        }
        if num_nulls:
            logger.info(
                f"[{name}] Numeric NaN remaining "
                f"(CatBoost handles natively ✓):"
            )
            for col, count in num_nulls.items():
                pct = count / len(df) * 100
                logger.info(f"  {col}: {count:,} ({pct:.1f}%)")

        logger.info(
            f"[{name}] Shape: {df.shape[0]:,} rows × {df.shape[1]:,} cols ✓"
        )

    if has_errors:
        raise RuntimeError(
            "CatBoost readiness validation failed — see errors above."
        )

    logger.info("CatBoost readiness validation passed ✓")


# ══════════════════════════════════════════════════════════
#  STEP 6 — SAVE CAT FEATURES LIST
# ══════════════════════════════════════════════════════════

def save_cat_features(cat_features: list[str]) -> None:
    """
    Saves categorical feature list to artifacts/catboost_cat_features.json.

    Why save this separately:
      - train_catboost.py loads this to pass to CatBoost Pool
      - Scoring pipeline loads this for real-time inference
      - Guarantees train/val/test use IDENTICAL cat_features list
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "cat_features":       cat_features,
        "cat_features_count": len(cat_features),
        "null_fill_value":    CAT_NULL_FILL,
    }

    with open(CAT_FEATURES_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"cat_features saved ({len(cat_features)} cols): "
        f"{CAT_FEATURES_PATH}"
    )


# ══════════════════════════════════════════════════════════
#  STEP 7 — SAVE CATBOOST-READY SPLITS
# ══════════════════════════════════════════════════════════

def save_catboost_splits(splits: dict[str, pd.DataFrame]) -> None:
    """
    Saves CatBoost-ready splits to data/versions/v2_catboost/.
    Creates directory if it doesn't exist.
    Writes a prep_manifest.json with shapes and dtype summary.
    """
    V2_DIR.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "train": V2_TRAIN,
        "val":   V2_VAL,
        "test":  V2_TEST,
    }

    manifest = {}
    for name, df in splits.items():
        out_path = out_paths[name]
        df.to_csv(out_path, index=False)

        manifest[name] = {
            "rows":       df.shape[0],
            "cols":       df.shape[1],
            "has_target": "isFraud" in df.columns,
            "dtypes": {
                "object_cols": df.select_dtypes("object").columns.tolist(),
                "float_cols":  df.select_dtypes("float64").columns.tolist(),
                "int_cols":    df.select_dtypes("int64").columns.tolist(),
            },
        }
        logger.info(
            f"Saved: {out_path} "
            f"({df.shape[0]:,} rows × {df.shape[1]:,} cols)"
        )

    manifest_path = V2_DIR / "prep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Prep manifest saved: {manifest_path}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def run_catboost_prep() -> list[str]:
    """
    Orchestrates all prep steps.
    Returns cat_features list for use by calling pipeline.
    """
    start = time.time()

    logger.info("=" * 60)
    logger.info("CATBOOST PREP — START")
    logger.info(f"Input  : {ENRICHED_DIR}")
    logger.info(f"Output : {V2_DIR}")
    logger.info("=" * 60)

    try:
        # Step 1 — Load enriched splits
        logger.info("\n[ Step 1 ] Loading enriched splits ...")
        splits = load_enriched_splits()

        # Step 2 — Identify categorical features (from train)
        # Use train to detect — consistent across all splits
        logger.info("\n[ Step 2 ] Identifying categorical features ...")
        cat_features = identify_cat_features(splits["train"])

        # Steps 3 & 4 — Fill cat NaN + cast binary flags (all splits)
        logger.info(
            "\n[ Steps 3-4 ] Filling categorical NaN + "
            "casting binary flags ..."
        )
        for name in SPLITS:
            splits[name] = fill_categorical_nulls(
                splits[name], cat_features, name
            )
            splits[name] = cast_binary_flags(splits[name], name)

        # Step 5 — Validate
        logger.info("\n[ Step 5 ] Validating CatBoost readiness ...")
        validate_catboost_ready(splits, cat_features)

        # Step 6 — Save cat features list
        logger.info("\n[ Step 6 ] Saving cat_features list ...")
        save_cat_features(cat_features)

        # Step 7 — Save splits
        logger.info("\n[ Step 7 ] Saving CatBoost-ready splits ...")
        save_catboost_splits(splits)

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"CATBOOST PREP — COMPLETE ({elapsed:.1f}s)")
        logger.info("=" * 60)

        return cat_features

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Prep error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    run_catboost_prep()