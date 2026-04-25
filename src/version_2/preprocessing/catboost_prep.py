# src/common/model_prep/catboost_prep.py
#
# Stage 2B — CatBoost Data Preparation
# Transforms enriched splits into CatBoost-ready format.
#
# FEATURE_MODE controls what features are used:
#   "all"   → ALL raw features + engineered features (DEFAULT)
#             Best for maximum signal — recommended for v2 rerun + v3
#   "top50" → Top 50 features from v1 XGBoost + engineered features
#             Use if you want to compare the top50 strategy
#
# Usage (from project ROOT):
#   python -m src.common.model_prep.catboost_prep

import sys
import json
import time
import traceback
from pathlib import Path

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
    V2_ALL_TRAIN,
    V2_ALL_VAL,
    V2_ALL_TEST,
)

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
#  ★ FEATURE MODE FLAG — change this before running ★
#
#  "all"   → all raw features + engineered (DEFAULT — recommended)
#  "top50" → top 50 features from v1 + engineered
# ══════════════════════════════════════════════════════════
FEATURE_MODE = "all"     # ← change to "top50" to use top-50 strategy


# ── Paths ──────────────────────────────────────────────────
ENRICHED_DIR      = ENRICHED
ARTIFACTS_DIR     = ARTIFACTS
CAT_FEATURES_PATH = ARTIFACTS_DIR / "catboost_cat_features.json"
TOP50_PATH        = ARTIFACTS_DIR / "top50_features.json"   # written by v1

# ── Output dirs: "all" mode vs "top50" mode ───────────────
# "all"   → data/versions/v2_catboost/       (same dir as before, fixed)
# "top50" → data/versions/v2_catboost_top50/ (separate — no overwrite)
OUTPUT_PATHS = {
    "all": {
        "train": V2_ALL_TRAIN,
        "val":   V2_ALL_VAL,
        "test":  V2_ALL_TEST,
    },
    "top50": {
        "train": V2_TRAIN,
        "val":   V2_VAL,
        "test":  V2_TEST,
    },
}

SPLITS      = ["train", "val", "test"]
CAT_NULL_FILL = "missing"

# ── Engineered feature columns (always included in both modes) ──
# These are additive — stacked on top of whatever raw features
# are selected. Never replace raw features.
ENGINEERED_COLS = [
    # Amount features
    "amt_log",
    "amt_cents",
    "amt_is_round",
    "amt_bucket",
    # Behavioral features
    "device_txn_count",
    "addr_is_unique",
    # Binary flags
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


# ══════════════════════════════════════════════════════════
#  STEP 1 — LOAD ENRICHED SPLITS
# ══════════════════════════════════════════════════════════

def load_enriched_splits() -> dict[str, pd.DataFrame]:
    """Loads enriched train/val/test from data/enriched/."""
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
#  STEP 2 — SELECT FEATURES BY MODE
# ══════════════════════════════════════════════════════════

def select_features(
    splits: dict[str, pd.DataFrame],
    mode: str,
) -> dict[str, pd.DataFrame]:
    """
    Selects feature columns based on FEATURE_MODE.

    MODE = "all":
      Keeps ALL columns from enriched splits.
      Engineered cols already exist in enriched — nothing to drop.
      This is the maximum-signal strategy. [web:430]
      CatBoost discovers feature interactions internally — more
      raw features = more combinations it can explore. [web:431]

    MODE = "top50":
      Loads top50 feature list from artifacts/top50_features.json
      (written by v1 evaluation pipeline).
      Keeps only those 50 cols + all engineered cols + target.
      Use this ONLY to compare strategies — not recommended as default.

    In both modes: target col (isFraud) always preserved in train/val.
    """
    if mode not in ("all", "top50"):
        raise ValueError(
            f"FEATURE_MODE must be 'all' or 'top50', got: '{mode}'"
        )

    logger.info(f"\n[ Feature Mode ] : {mode.upper()}")

    if mode == "all":
        logger.info(
            "  Using ALL raw features + engineered features.\n"
            f"  Total cols in train: {splits['train'].shape[1]}"
        )
        # Nothing to drop — enriched already has everything
        return splits

    # ── mode == "top50" ────────────────────────────────────
    if not TOP50_PATH.exists():
        raise FileNotFoundError(
            f"top50_features.json not found: {TOP50_PATH}\n"
            f"This is written by the v1 evaluation pipeline.\n"
            f"Run v1 evaluate first, or switch FEATURE_MODE to 'all'."
        )

    with open(TOP50_PATH, "r") as f:
        top50_data = json.load(f)

    top50_raw = top50_data["features"]   # list of 50 feature names
    logger.info(f"  Loaded {len(top50_raw)} top features from v1 model")

    # Build final feature set: top50 raw + engineered + target
    for name, df in splits.items():
        target_cols = ["isFraud"] if "isFraud" in df.columns else []

        # Filter top50 to only cols that exist in this split
        available_raw = [c for c in top50_raw if c in df.columns]
        missing_raw   = set(top50_raw) - set(available_raw)
        if missing_raw:
            logger.warning(
                f"  [{name}] {len(missing_raw)} top50 cols not found "
                f"in split: {sorted(missing_raw)}"
            )

        # Engineered cols that exist in this split
        available_eng = [c for c in ENGINEERED_COLS if c in df.columns]

        # Final column set — deduplicated, target last
        keep_cols = list(dict.fromkeys(
            available_raw + available_eng + target_cols
        ))

        splits[name] = df[keep_cols]
        logger.info(
            f"  [{name}] Selected {len(keep_cols)} cols "
            f"({len(available_raw)} raw + "
            f"{len(available_eng)} engineered"
            + (f" + 1 target)" if target_cols else ")")
        )

    return splits


# ══════════════════════════════════════════════════════════
#  STEP 3 — IDENTIFY CATEGORICAL COLUMNS
# ══════════════════════════════════════════════════════════

def identify_cat_features(df: pd.DataFrame) -> list[str]:
    """Detects categorical columns by dtype. Excludes isFraud."""
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
#  STEP 4 — FILL CATEGORICAL NaN
# ══════════════════════════════════════════════════════════

def fill_categorical_nulls(
    df: pd.DataFrame,
    cat_features: list[str],
    split_name: str,
) -> pd.DataFrame:
    """Fills NaN in categorical columns with 'missing'."""
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
            df[col] = df[col].astype(str)

    return df


# ══════════════════════════════════════════════════════════
#  STEP 5 — CAST BINARY FLAG COLUMNS TO INT
# ══════════════════════════════════════════════════════════

def cast_binary_flags(
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Ensures binary flag columns are int dtype."""
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
        "amt_is_round",    # added — engineered binary flag
    ]

    cast_count = 0
    for col in BINARY_FLAG_COLS:
        if col not in df.columns:
            continue
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
#  STEP 6 — VALIDATE CATBOOST READINESS
# ══════════════════════════════════════════════════════════

def validate_catboost_ready(
    splits: dict[str, pd.DataFrame],
    cat_features: list[str],
) -> None:
    """Final validation before saving."""
    logger.info("Validating CatBoost readiness ...")
    has_errors = False

    for name, df in splits.items():
        for col in cat_features:
            if col not in df.columns:
                continue
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.error(
                    f"[{name}] Categorical col '{col}' still has "
                    f"{null_count:,} NaN — CatBoost will error"
                )
                has_errors = True
            if df[col].dtype not in ["object", "string"]:
                logger.error(
                    f"[{name}] Categorical col '{col}' dtype is "
                    f"'{df[col].dtype}' — expected string/object"
                )
                has_errors = True

        if name in ["train", "val"] and "isFraud" not in df.columns:
            logger.error(f"[{name}] isFraud missing — cannot train")
            has_errors = True
        if name == "test" and "isFraud" in df.columns:
            logger.error("[test] isFraud found in test split")
            has_errors = True

        num_cols  = df.select_dtypes(include=["float64", "int64"]).columns
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
#  STEP 7 — SAVE CAT FEATURES LIST
# ══════════════════════════════════════════════════════════

def save_cat_features(cat_features: list[str], mode: str) -> None:
    """Saves categorical feature list + mode used to artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "cat_features":       cat_features,
        "cat_features_count": len(cat_features),
        "null_fill_value":    CAT_NULL_FILL,
        "feature_mode":       mode,   # ← added for traceability
    }

    with open(CAT_FEATURES_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"cat_features saved ({len(cat_features)} cols): "
        f"{CAT_FEATURES_PATH}"
    )


# ══════════════════════════════════════════════════════════
#  STEP 8 — SAVE CATBOOST-READY SPLITS
# ══════════════════════════════════════════════════════════

def save_catboost_splits(
    splits: dict[str, pd.DataFrame],
    mode: str,
) -> None:
    """
    Saves CatBoost-ready splits to the correct output dir.

    mode="all"   → data/versions/v2_catboost_all/
    mode="top50" → data/versions/v2_catboost_top50/

    Separate dirs guarantee the two strategies never overwrite
    each other, so you can compare results at any time.
    """
    out_paths = OUTPUT_PATHS[mode]
    out_dir   = out_paths["train"].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, df in splits.items():
        out_path = out_paths[name]
        df.to_csv(out_path, index=False)

        manifest[name] = {
            "rows":         df.shape[0],
            "cols":         df.shape[1],
            "has_target":   "isFraud" in df.columns,
            "feature_mode": mode,
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

    manifest_path = out_dir / "prep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Prep manifest saved: {manifest_path}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def run_catboost_prep(mode: str = FEATURE_MODE) -> list[str]:
    """
    Orchestrates all prep steps.
    Returns cat_features list for use by calling pipeline.

    Args:
        mode: "all" (default) or "top50"
              Override FEATURE_MODE at the function call level
              so v2_pipeline.py can pass it in explicitly.
    """
    start = time.time()

    if mode not in ("all", "top50"):
        raise ValueError(
            f"mode must be 'all' or 'top50', got '{mode}'"
        )

    out_dir = OUTPUT_PATHS[mode]["train"].parent

    logger.info("=" * 60)
    logger.info("CATBOOST PREP — START")
    logger.info(f"Feature mode : {mode.upper()}")
    logger.info(f"Input        : {ENRICHED_DIR}")
    logger.info(f"Output       : {out_dir}")
    logger.info("=" * 60)

    try:
        # Step 1 — Load
        logger.info("\n[ Step 1 ] Loading enriched splits ...")
        splits = load_enriched_splits()

        # Step 2 — Feature selection by mode
        logger.info("\n[ Step 2 ] Selecting features ...")
        splits = select_features(splits, mode)

        # Step 3 — Identify cat features (from train after selection)
        logger.info("\n[ Step 3 ] Identifying categorical features ...")
        cat_features = identify_cat_features(splits["train"])

        # Steps 4 & 5 — Fill cat NaN + cast binary flags
        logger.info(
            "\n[ Steps 4-5 ] Filling categorical NaN + "
            "casting binary flags ..."
        )
        for name in SPLITS:
            splits[name] = fill_categorical_nulls(
                splits[name], cat_features, name
            )
            splits[name] = cast_binary_flags(splits[name], name)

        # Step 6 — Validate
        logger.info("\n[ Step 6 ] Validating CatBoost readiness ...")
        validate_catboost_ready(splits, cat_features)

        # Step 7 — Save cat features
        logger.info("\n[ Step 7 ] Saving cat_features list ...")
        save_cat_features(cat_features, mode)

        # Step 8 — Save splits
        logger.info("\n[ Step 8 ] Saving CatBoost-ready splits ...")
        save_catboost_splits(splits, mode)

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(
            f"CATBOOST PREP — COMPLETE ({elapsed:.1f}s) | "
            f"mode={mode.upper()}"
        )
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