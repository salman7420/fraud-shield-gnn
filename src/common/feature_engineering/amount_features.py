# src/common/feature_engineering/amount_features.py

from src.utils.logger import get_logger
import numpy as np
import pandas as pd
from typing import Optional

logger = get_logger(__name__)

AMT_VERY_LOW_THRESHOLD = 26.0
AMT_HIGH_THRESHOLD     = 280.0
CHARM_CENTS            = {0.95}


def _add_static_amount_features(df: pd.DataFrame,
                                amt_col: str) -> pd.DataFrame:
    """
    Pure math transforms — no fitting needed.
    Called by both fit_amount_features and apply_amount_features.
    """
    amt = df[amt_col]

    df["amt_log"]         = np.log1p(amt)
    df["amt_cents"]       = (amt % 1).round(2)

    df["amt_cents_type"]  = 2                                        # default: other
    df.loc[df["amt_cents"] == 0.00, "amt_cents_type"] = 0            # round
    df.loc[df["amt_cents"].isin(CHARM_CENTS), "amt_cents_type"] = 1  # charm

    df["amt_is_very_low"] = (amt < AMT_VERY_LOW_THRESHOLD).astype(int)
    df["amt_is_high"]     = (amt > AMT_HIGH_THRESHOLD).astype(int)

    return df


def fit_amount_features(train_df: pd.DataFrame,
                        amt_col: str = "TransactionAmt") -> dict:
    """
    STEP 1 — Call on train ONLY.
    Learns the decile bin edges from train and returns them as params.
    These params must be passed to apply_amount_features for val/test.
    """
    if amt_col not in train_df.columns:
        logger.warning(f"'{amt_col}' not found — skipping amount features fit")
        return {}

    _, bin_edges = pd.qcut(
        train_df[amt_col],
        q=10,
        retbins=True,
        duplicates="drop"
    )

    # Extend edges to -inf/+inf so val/test values outside train's
    # min/max don't become NaN (e.g. a $32k transaction in test)
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf

    params = {"amt_bucket_bins": bin_edges}
    logger.info(f"Amount feature params fitted — bucket edges: {bin_edges.round(2)}")
    return params


def apply_amount_features(df: pd.DataFrame,
                          params: dict,
                          amt_col: str = "TransactionAmt") -> pd.DataFrame:
    """
    STEP 2 — Call on train, val, AND test using the SAME params.
    Applies static transforms + the pre-fitted bucket bins.
    """
    if amt_col not in df.columns:
        logger.warning(f"'{amt_col}' not found — skipping amount features")
        return df

    # Static features — pure math, no leakage risk
    df = _add_static_amount_features(df, amt_col)

    # Bucketing — use ONLY the edges learned from train
    bin_edges = params.get("amt_bucket_bins")
    if bin_edges is not None:
        df["amt_bucket"] = pd.cut(
            df[amt_col],
            bins=bin_edges,
            labels=False,       # 0–9 integer labels
            include_lowest=True
        )
    else:
        logger.warning("No bucket bins found in params — amt_bucket skipped")

    logger.info("Amount features applied: amt_log, amt_cents, amt_cents_type, "
                "amt_is_very_low, amt_is_high, amt_bucket")
    return df


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "TransactionAmt": [50.00, 49.95, 117.47, 0.251, 31937.391,
                           15.00, 300.00, 80.99]
    })
    # Simulate the real pipeline flow
    params = fit_amount_features(test_df)          # fit on "train"
    result = apply_amount_features(test_df, params) # apply to same
    print(result[["TransactionAmt", "amt_log", "amt_cents",
                  "amt_cents_type", "amt_is_very_low",
                  "amt_is_high", "amt_bucket"]])