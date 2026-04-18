""""
This file contains base functions to create the base dataset that will extend to different versions for different models.
Fucntions:

- Load Data
- Merge Data
- Remove columns with more than 80% of data
- Save base data 

"""

import pandas as pd
from src.utils.data_configs import (
    RAW_TRANSACTION,
    RAW_IDENTITY,
    BASE_DATA,
    NULL_THRESHOLD
)

from src.data_ingestion.load_data import load_data, save_data
from utils.logger import get_logger

logger = get_logger(__name__) 

def merge_data(
    df_transaction: pd.DataFrame,
    df_identity: pd.DataFrame
) -> pd.DataFrame:
    
     """
    Left join transaction and identity data on TransactionID.
    All transactions are kept; identity columns filled with NaN where missing.

    Args:
        df_transaction: Transaction DataFrame
        df_identity:    Identity DataFrame

    Returns:
        Merged DataFrame
    """
     
     logger.info("Merging transaction and identity datasets...")

     if "TransactionID" not in df_transaction.columns:
        raise KeyError("'TransactionID' column missing from transaction data")

     if "TransactionID" not in df_identity.columns:
        raise KeyError("'TransactionID' column missing from identity data")
     
     df_merged = df_transaction.merge(
        df_identity,
        on="TransactionID",
        how="left"
    )

     logger.info(
        f"Merge complete | "
        f"Transactions: {len(df_transaction):,} | "
        f"Matched identity rows: {df_identity['TransactionID'].isin(df_transaction['TransactionID']).sum():,} | "
        f"Merged shape: {df_merged.shape}"
    )
     
     return df_merged


def remove_high_null_columns(
    df: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
    """
    TODO: Remove columns where null percentage exceeds the threshold.

    Args:
        df:        Input DataFrame
        threshold: Null percentage threshold (default from config = 0.80)

    Returns:
        DataFrame with high-null columns removed
    """
    # ── Placeholder — to be implemented ──────────────────
    logger.info(f"[TODO] Null column removal (threshold={threshold}) — skipping for now")
    return df


def build_base() -> pd.DataFrame:
    """
    Full pipeline to build the base dataset:
        1. Load raw transaction and identity data
        2. Merge on TransactionID
        3. Remove columns with > null_threshold missing values  [TODO]
        4. Save to BASE_DATA path

    Args:
        null_threshold: Columns with null % above this are dropped (default 0.80)

    Returns:
        Base DataFrame ready for versioning
    """
    logger.info("=" * 50)
    logger.info("Starting base dataset build pipeline")
    logger.info("=" * 50)

    # ── Step 1: Load ──────────────────────────────────────
    logger.info("Step 1/4 — Loading raw data")
    df_transaction = load_data(RAW_TRANSACTION)
    df_identity    = load_data(RAW_IDENTITY)

    # ── Step 2: Merge ─────────────────────────────────────
    logger.info("Step 2/4 — Merging datasets")
    df = merge_data(df_transaction, df_identity)

    # ── Step 3: Remove high-null columns ──────────────────
    logger.info("Step 3/4 — Removing high-null columns")
    df = remove_high_null_columns(df, threshold=NULL_THRESHOLD)

    # ── Step 4: Save ──────────────────────────────────────
    logger.info("Step 4/4 — Saving base dataset")
    save_data(df, BASE_DATA)

    logger.info("=" * 50)
    logger.info(f"Base dataset build complete")
    logger.info("=" * 50)

    return df


if __name__ == "__main__":
    # Run directly from terminal: python -m src.data_processing.base_builder
    df_base = build_base()
    print(df_base.head())