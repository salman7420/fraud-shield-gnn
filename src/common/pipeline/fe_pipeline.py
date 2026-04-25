# src/common/enriched_pipeline/build_enriched.py
#
# Stage 1 — Common Feature Engineering Pipeline
# Runs once, produces enriched splits used by both XGBoost and CatBoost.
#
# Usage (from project ROOT):
#   python -m src.common.enriched_pipeline.build_enriched
#
# Output:
#   data/enriched/train.csv
#   data/enriched/val.csv
#   data/enriched/test.csv
#   artifacts/fe_params.pkl

import os
import sys
import json
import pickle
import time
import traceback
from pathlib import Path

from src.utils.data_configs import V1_SHAP, ARTIFACTS, ENRICHED
import numpy as np
import pandas as pd

# ── Project root on path ───────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.common.feature_store.feature_registry import (
    ALL_ENGINEERED_FEATURES,
    TIME_FEATURES,
    AMOUNT_FEATURES,
    NULL_FLAG_FEATURES,
    AGGREGATION_FEATURES,
    RATIO_FEATURES,
    GRAPH_FEATURES,
)
from src.common.feature_engineering.time_features        import add_time_features
from src.common.feature_engineering.amount_features      import (
    fit_amount_features, apply_amount_features,
)
from src.common.feature_engineering.null_flags           import (
    fit_null_flags, apply_null_flags,
)
from src.common.feature_engineering.aggregation_features import (
    fit_aggregation_features, apply_aggregation_features,
)
from src.common.feature_engineering.ratio_features       import add_ratio_features
from src.common.feature_engineering.graph_features       import (
    fit_graph_features, apply_graph_features,
)

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════
BASE_DIR      = ROOT / "data" / "base"
ENRICHED_DIR  = ENRICHED
ARTIFACTS_DIR = ARTIFACTS
TOP_FEATURES_PATH = V1_SHAP / "top_features.json"

SPLITS = ["train", "val", "test"]


# ══════════════════════════════════════════════════════════
#  STEP 1 — LOAD
# ══════════════════════════════════════════════════════════

def load_base_splits() -> dict[str, pd.DataFrame]:
    """
    Loads train, val, test from data/base/.
    Validates files exist and are non-empty before loading.
    Returns dict: {"train": df, "val": df, "test": df}
    """
    splits = {}
    for name in SPLITS:
        path = BASE_DIR / f"{name}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Base split not found: {path}\n"
                f"Expected all 3 splits in: {BASE_DIR}"
            )

        if path.stat().st_size == 0:
            raise ValueError(f"Base split is empty: {path}")

        logger.info(f"Loading {name}.csv ...")
        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"Loaded dataframe is empty: {path}")

        logger.info(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]:,} cols")
        splits[name] = df

    return splits


# ══════════════════════════════════════════════════════════
#  STEP 2 — VALIDATE INPUT COLUMNS
# ══════════════════════════════════════════════════════════

# Minimum columns required for feature engineering to run
REQUIRED_COLS = {
    "TransactionDT",   # time features
    "TransactionAmt",  # amount features + ratios
    "card1",           # aggregation + graph features
    "P_emaildomain",   # aggregation + graph features
    "isFraud",         # aggregation target encoding (train only)
}

REQUIRED_COLS_TEST = REQUIRED_COLS - {"isFraud"}


def validate_columns(splits: dict[str, pd.DataFrame]) -> None:
    """
    Checks that all required columns exist in each split.
    Logs warnings for optional columns that are missing
    (DeviceInfo, addr1 — used in graph features but nullable).
    Raises ValueError if a hard-required column is missing.
    """
    optional_cols = {"DeviceInfo", "addr1", "DeviceType"}

    for name, df in splits.items():
        required = REQUIRED_COLS_TEST if name == "test" else REQUIRED_COLS
        missing_required = required - set(df.columns)
        missing_optional = optional_cols - set(df.columns)

        if missing_required:
            raise ValueError(
                f"Split '{name}' is missing required columns: "
                f"{sorted(missing_required)}"
            )

        if missing_optional:
            logger.warning(
                f"Split '{name}' missing optional columns "
                f"{sorted(missing_optional)} — "
                f"related graph features will be NaN"
            )

    logger.info("Column validation passed ✓")


# ══════════════════════════════════════════════════════════
#  STEP 3-8 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def run_feature_engineering(
    splits: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Runs all 6 feature engineering steps in dependency order.
    Returns enriched splits dict + combined fe_params dict.

    Fit order — all fit on train only:
      time        → pure transform (no fit needed)
      amount      → fit: bucket bin edges
      null_flags  → fit: which cols to flag
      aggregation → fit: card1/email lookup tables
      ratio       → pure transform (depends on agg cols)
      graph       → fit: device/addr/pair lookup tables

    Leakage protection:
      Every fit_*() call receives ONLY splits["train"]
      Every apply_*() call receives the same params dict
    """
    train = splits["train"]
    fe_params = {}

    # ── Step 3: Time Features (pure transform) ─────────────
    logger.info("─" * 50)
    logger.info("Step 3/8 — Time Features (pure transform)")
    for name in SPLITS:
        splits[name] = add_time_features(splits[name])
    _verify_features_added(splits, TIME_FEATURES, "time")

    # ── Step 4: Amount Features ────────────────────────────
    logger.info("─" * 50)
    logger.info("Step 4/8 — Amount Features (fit on train)")
    amt_params = fit_amount_features(train)
    fe_params.update(amt_params)
    for name in SPLITS:
        splits[name] = apply_amount_features(splits[name], amt_params)
    _verify_features_added(splits, AMOUNT_FEATURES, "amount")

    # ── Step 5: Null Flags ─────────────────────────────────
    logger.info("─" * 50)
    logger.info("Step 5/8 — Null Flags (fit on train)")
    null_params = fit_null_flags(train)
    fe_params.update(null_params)
    for name in SPLITS:
        splits[name] = apply_null_flags(splits[name], null_params)
    _verify_features_added(splits, NULL_FLAG_FEATURES, "null_flags")

    # ── Step 6: Aggregation Features ──────────────────────
    logger.info("─" * 50)
    logger.info("Step 6/8 — Aggregation Features (fit on train)")
    agg_params = fit_aggregation_features(train)
    fe_params.update(agg_params)
    for name in SPLITS:
        splits[name] = apply_aggregation_features(splits[name], agg_params)
    _verify_features_added(splits, AGGREGATION_FEATURES, "aggregation")

    # ── Step 7: Ratio Features (pure transform) ────────────
    # Must come AFTER aggregation — needs card1_mean_amt, card1_std_amt
    logger.info("─" * 50)
    logger.info("Step 7/8 — Ratio Features (pure transform, needs agg cols)")
    for name in SPLITS:
        splits[name] = add_ratio_features(splits[name])
    _verify_features_added(splits, RATIO_FEATURES, "ratio")

    # ── Step 8: Graph Features ─────────────────────────────
    logger.info("─" * 50)
    logger.info("Step 8/8 — Graph Features (fit on train)")
    graph_params = fit_graph_features(train)
    fe_params.update(graph_params)
    for name in SPLITS:
        splits[name] = apply_graph_features(splits[name], graph_params)
    _verify_features_added(splits, GRAPH_FEATURES, "graph")

    logger.info("─" * 50)
    logger.info(
        f"Feature engineering complete ✓ — "
        f"28 engineered features added to all splits"
    )
    return splits, fe_params


def _verify_features_added(
    splits: dict[str, pd.DataFrame],
    expected_features: list[str],
    block_name: str,
) -> None:
    """
    After each FE step, confirms every expected feature
    was actually added to all splits. Raises if any are missing.
    """
    for name, df in splits.items():
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            raise RuntimeError(
                f"[{block_name}] Features missing from '{name}' split "
                f"after engineering: {missing}"
            )


# ══════════════════════════════════════════════════════════
#  STEP 9 — LOAD TOP FEATURES
# ══════════════════════════════════════════════════════════

def load_top_features() -> list[str]:
    """
    Loads SHAP top-50 raw feature names from version_1 results.
    Handles JSON format: {"top_features": ["C13", "TransactionDT", ...]}

    Falls back to keeping ALL raw columns if file not found.
    """
    if not TOP_FEATURES_PATH.exists():
        logger.warning(
            f"top_features.json not found at {TOP_FEATURES_PATH} — "
            f"keeping all raw columns (Mode B fallback). "
            f"Re-run after first model to enable column filtering."
        )
        return []

    with open(TOP_FEATURES_PATH, "r") as f:
        data = json.load(f)

    # ── Handle all known formats ───────────────────────────
    # Format A (your format): {"top_features": ["C13", ...]}
    if isinstance(data, dict):
        # Try known wrapper keys
        for key in ["top_features", "features", "columns", "selected_features"]:
            if key in data:
                top_features = data[key]
                logger.info(
                    f"Loaded {len(top_features)} top features "
                    f"from key '{key}' in {TOP_FEATURES_PATH.name}"
                )
                break
        else:
            raise ValueError(
                f"top_features.json is a dict but none of the expected keys "
                f"['top_features', 'features', 'columns'] were found. "
                f"Keys present: {list(data.keys())}"
            )

    # Format B (plain list): ["C13", "TransactionDT", ...]
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValueError("top_features.json contains an empty list.")
        if isinstance(data[0], str):
            top_features = data
            logger.info(
                f"Loaded {len(top_features)} top features "
                f"(plain list format) from {TOP_FEATURES_PATH.name}"
            )
        elif isinstance(data[0], dict):
            # Format C (SHAP dict list): [{"feature": "C13", ...}, ...]
            for key in ["feature", "name", "col", "column"]:
                if key in data[0]:
                    top_features = [d[key] for d in data]
                    logger.info(
                        f"Loaded {len(top_features)} top features "
                        f"(dict list, key='{key}') from {TOP_FEATURES_PATH.name}"
                    )
                    break
            else:
                raise ValueError(
                    f"top_features.json list-of-dicts format unrecognised. "
                    f"Keys found in first item: {list(data[0].keys())}"
                )
        else:
            raise ValueError(
                f"top_features.json list contains unexpected type: "
                f"{type(data[0])}. Expected str or dict."
            )
    else:
        raise ValueError(
            f"top_features.json root must be a dict or list. "
            f"Got: {type(data)}"
        )

    # ── Validate all entries are strings ───────────────────
    if not all(isinstance(f, str) for f in top_features):
        raise ValueError(
            "top_features.json contains non-string feature names."
        )

    logger.info(f"Top features: {top_features}")
    return top_features


# ══════════════════════════════════════════════════════════
#  STEP 10 — SELECT FINAL COLUMNS
# ══════════════════════════════════════════════════════════

def select_final_columns(
    splits: dict[str, pd.DataFrame],
    top_raw_features: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Keeps only: top_raw_features + ALL_ENGINEERED_FEATURES + isFraud.
    Drops all other raw columns (V cols, C cols, id cols not in top-50).

    If top_raw_features is empty (Mode B fallback):
      → Keeps all raw columns + engineered features.

    Handles:
      - top_raw_features that don't exist in df (warns, skips)
      - test split has no isFraud column
      - engineered features not in df (shouldn't happen after Step 3-8)
    """
    for name, df in splits.items():
        if top_raw_features:
            # Only keep raw features that actually exist in this split
            available_raw = [
                f for f in top_raw_features if f in df.columns
            ]
            missing_raw = set(top_raw_features) - set(available_raw)
            if missing_raw:
                logger.warning(
                    f"[{name}] {len(missing_raw)} top raw features not "
                    f"found in split: {sorted(missing_raw)}"
                )
        else:
            # Mode B — keep all raw columns
            available_raw = [
                c for c in df.columns
                if c not in ALL_ENGINEERED_FEATURES
                and c != "isFraud"
            ]

        # Always include engineered features that exist
        available_engineered = [
            f for f in ALL_ENGINEERED_FEATURES if f in df.columns
        ]
        missing_engineered = set(ALL_ENGINEERED_FEATURES) - set(available_engineered)
        if missing_engineered:
            logger.warning(
                f"[{name}] Engineered features missing from split: "
                f"{sorted(missing_engineered)}"
            )

        # Build final column list
        keep = available_raw + available_engineered

        # Add target for train/val only
        if name == "test":
            if "isFraud" in df.columns:
                logger.info(
                    "[test] isFraud detected and removed — "
                    "test split saved without target column"
                )
            # do NOT add isFraud to keep — it stays excluded
        else:
            if "isFraud" in df.columns:
                keep = keep + ["isFraud"]

        splits[name] = df[keep]

        logger.info(
            f"[{name}] Final shape: "
            f"{splits[name].shape[0]:,} rows × "
            f"{splits[name].shape[1]:,} cols "
            f"({len(available_raw)} raw + "
            f"{len(available_engineered)} engineered"
            + (" + 1 target)" if "isFraud" in splits[name].columns else ")")
        )

    return splits


# ══════════════════════════════════════════════════════════
#  STEP 11 — VALIDATE OUTPUT
# ══════════════════════════════════════════════════════════

def validate_output(splits: dict[str, pd.DataFrame]) -> None:
    """
    Post-engineering checks:
      1. Row counts unchanged from input
      2. All 28 engineered features present
      3. isFraud present in train/val, absent in test
      4. No all-null engineered feature columns
      5. Engineered feature value range spot checks
    """
    logger.info("Running output validation ...")
    has_errors = False

    for name, df in splits.items():

        # Check 1: engineered features all present
        missing_fe = [
            f for f in ALL_ENGINEERED_FEATURES if f not in df.columns
        ]
        if missing_fe:
            logger.error(
                f"[{name}] Missing engineered features: {missing_fe}"
            )
            has_errors = True

        # Check 2: isFraud presence
        if name in ["train", "val"] and "isFraud" not in df.columns:
            logger.error(f"[{name}] isFraud column missing")
            has_errors = True

        if name == "test" and "isFraud" in df.columns:
            logger.error(
                "[test] isFraud still present after column selection — "
                "select_final_columns() did not drop it correctly"
            )
            has_errors = True

        # Check 3: no fully-null engineered feature
        for feat in ALL_ENGINEERED_FEATURES:
            if feat in df.columns and df[feat].isna().all():
                logger.error(
                    f"[{name}] Engineered feature '{feat}' is entirely NaN"
                )
                has_errors = True

        # Check 4: value range spot checks
        if "hour" in df.columns:
            bad = df["hour"].dropna()
            if not bad.between(0, 23).all():
                logger.error(f"[{name}] 'hour' has values outside 0-23")
                has_errors = True

        if "amt_log" in df.columns:
            if (df["amt_log"].dropna() < 0).any():
                logger.error(f"[{name}] 'amt_log' has negative values")
                has_errors = True

        if "email_fraud_rate" in df.columns:
            bad_rates = df["email_fraud_rate"].dropna()
            if not bad_rates.between(0, 1).all():
                logger.error(
                    f"[{name}] 'email_fraud_rate' has values outside [0,1]"
                )
                has_errors = True

        # Check 5: null report for engineered features
        null_counts = {
            f: int(df[f].isna().sum())
            for f in ALL_ENGINEERED_FEATURES
            if f in df.columns and df[f].isna().sum() > 0
        }
        if null_counts:
            logger.info(
                f"[{name}] Engineered feature null counts "
                f"(expected — XGBoost/CatBoost handle natively):"
            )
            for feat, count in null_counts.items():
                pct = count / len(df) * 100
                logger.info(f"  {feat}: {count:,} ({pct:.1f}%)")

    if has_errors:
        raise RuntimeError(
            "Output validation failed — see errors above. "
            "Fix before saving enriched data."
        )

    logger.info("Output validation passed ✓")


# ══════════════════════════════════════════════════════════
#  STEP 12 — SAVE ENRICHED SPLITS
# ══════════════════════════════════════════════════════════

def save_enriched_splits(splits: dict[str, pd.DataFrame]) -> None:
    """
    Saves enriched train/val/test to data/enriched/.
    Creates directory if it doesn't exist.
    Writes a manifest.json with shapes and column lists.
    """
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}

    for name, df in splits.items():
        out_path = ENRICHED_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)

        manifest[name] = {
            "rows":    df.shape[0],
            "cols":    df.shape[1],
            "columns": df.columns.tolist(),
        }
        logger.info(
            f"Saved: {out_path} "
            f"({df.shape[0]:,} rows × {df.shape[1]:,} cols)"
        )

    # Save manifest for traceability
    manifest_path = ENRICHED_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved: {manifest_path}")


# ══════════════════════════════════════════════════════════
#  STEP 13 — SAVE FE PARAMS
# ══════════════════════════════════════════════════════════

def save_fe_params(fe_params: dict) -> None:
    """
    Saves all fitted FE params to artifacts/fe_params.pkl.
    Used for:
      - Real-time scoring (load params, call apply_*())
      - Reproducing enriched data exactly
      - Debugging feature values

    Also saves a human-readable summary (fe_params_summary.json)
    with key counts but not the full lookup dicts (too large).
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Full params pickle (for pipeline use)
    pkl_path = ARTIFACTS_DIR / "fe_params.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(fe_params, f)
    logger.info(f"FE params saved: {pkl_path}")

    # Human-readable summary (for debugging)
    summary = {}
    for key, val in fe_params.items():
        if isinstance(val, dict):
            summary[key] = f"dict with {len(val):,} entries"
        elif isinstance(val, (list, np.ndarray)):
            summary[key] = f"array/list with {len(val)} elements"
        elif isinstance(val, float):
            summary[key] = round(val, 6)
        else:
            summary[key] = str(val)

    summary_path = ARTIFACTS_DIR / "fe_params_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"FE params summary saved: {summary_path}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main() -> None:
    start = time.time()

    logger.info("=" * 60)
    logger.info("BUILD ENRICHED PIPELINE — START")
    logger.info(f"Project root : {ROOT}")
    logger.info(f"Base data    : {BASE_DIR}")
    logger.info(f"Output       : {ENRICHED_DIR}")
    logger.info(f"Artifacts    : {ARTIFACTS_DIR}")
    logger.info("=" * 60)

    try:
        # Step 1 — Load
        logger.info("\n[ Step 1 ] Loading base splits ...")
        splits = load_base_splits()

        # Step 2 — Validate input
        logger.info("\n[ Step 2 ] Validating input columns ...")
        validate_columns(splits)

        # Steps 3–8 — Feature Engineering
        logger.info("\n[ Steps 3–8 ] Running feature engineering ...")
        splits, fe_params = run_feature_engineering(splits)

        # Step 9 — Load top features
        logger.info("\n[ Step 9 ] Loading SHAP top features ...")
        #top_raw_features = load_top_features()
        top_raw_features = []
        logger.info("Mode B — keeping ALL raw columns + 28 engineered features")


        # Step 10 — Select final columns
        logger.info("\n[ Step 10 ] Selecting final columns ...")
        splits = select_final_columns(splits, top_raw_features)

        # Step 11 — Validate output
        logger.info("\n[ Step 11 ] Validating output ...")
        validate_output(splits)

        # Step 12 — Save enriched splits
        logger.info("\n[ Step 12 ] Saving enriched splits ...")
        save_enriched_splits(splits)

        # Step 13 — Save FE params
        logger.info("\n[ Step 13 ] Saving FE params ...")
        save_fe_params(fe_params)

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"BUILD ENRICHED PIPELINE — COMPLETE ({elapsed:.1f}s)")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)

    except RuntimeError as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()