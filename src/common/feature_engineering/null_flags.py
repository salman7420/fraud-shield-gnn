# src/common/feature_engineering/null_flags.py

from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

# ── EDA findings summary ───────────────────────────────────
# id_01 null = ALL other id cols null → correlation = 1.000
# id_13 and id_16 deviate from the group (corr 0.83–0.91)
# R_emaildomain, DeviceType, DeviceInfo corr > 0.88 with id_01 null
#   → fully captured by id_data_present, no separate flag needed
#
# Fraud null rates vs legit:
#   id_data_present: fraud=45%  legit=75%  diff=0.30  ← primary signal
#   id_13_was_null:  fraud=49%  legit=78%  diff=0.29  ← independent
#   id_16_was_null:  fraud=51%  legit=77%  diff=0.27  ← independent


def fit_null_flags(train_df: pd.DataFrame) -> dict:
    """
    STEP 1 — Call on train ONLY.
    Confirms the expected null-flag columns exist in the dataset.
    Returns params dict for apply_null_flags.
    """
    required_cols = ["id_01", "id_13", "id_16"]
    missing = [c for c in required_cols if c not in train_df.columns]

    if missing:
        logger.warning(f"Null flag columns missing from dataset: {missing}")

    params = {
        "null_flag_cols": [c for c in required_cols if c in train_df.columns]
    }

    logger.info(f"Null flags fitted — will flag: {params['null_flag_cols']}")
    return params


def apply_null_flags(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    STEP 2 — Call on train, val, AND test using the SAME params.

    Creates 3 features:
      id_data_present  — 1 if device identity data exists (covers 19+ cols)
      id_13_was_null   — 1 if id_13 is null (independent null behavior)
      id_16_was_null   — 1 if id_16 is null (independent null behavior)
    """
    null_flag_cols = params.get("null_flag_cols", [])

    if "id_01" in null_flag_cols:
        df["id_data_present"] = df["id_01"].notnull().astype(int)
        logger.info(
            f"id_data_present: "
            f"{df['id_data_present'].sum():,} rows have device data "
            f"({df['id_data_present'].mean()*100:.1f}%)"
        )

    if "id_13" in null_flag_cols:
        df["id_13_was_null"] = df["id_13"].isnull().astype(int)

    if "id_16" in null_flag_cols:
        df["id_16_was_null"] = df["id_16"].isnull().astype(int)

    logger.info("Null flags applied: id_data_present, id_13_was_null, id_16_was_null")
    return df


if __name__ == "__main__":
    import numpy as np

    # Simulate 6 rows: 3 with device data, 3 without
    test_df = pd.DataFrame({
        "id_01": [1.0, 2.0, np.nan, np.nan, 1.0, np.nan],
        "id_13": [1.0, np.nan, np.nan, 1.0,  np.nan, np.nan],
        "id_16": [1.0, 1.0,   np.nan, np.nan, 1.0, np.nan],
    })

    params = fit_null_flags(test_df)
    result = apply_null_flags(test_df, params)
    print(result[["id_01", "id_13", "id_16",
                  "id_data_present", "id_13_was_null", "id_16_was_null"]])