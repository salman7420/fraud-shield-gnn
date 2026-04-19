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
from src.utils.logger import get_logger

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
    threshold: float = NULL_THRESHOLD
) -> pd.DataFrame:
    """
    Remove columns where the percentage of missing values exceeds the threshold.

    Args:
        df:        Input DataFrame
        threshold: Drop columns with null % above this value (default 0.80 = 80%)

    Returns:
        DataFrame with high-null columns removed

    Raises:
        ValueError: If threshold is not between 0 and 1
        TypeError:  If df is not a DataFrame
    """

    # ── Input validation ──────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")

    if not 0 < threshold < 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    if df.empty:
        logger.warning("Received an empty DataFrame — skipping null column removal")
        return df

    # ── Calculate null percentage per column ──────────────
    null_pct = df.isnull().mean()  # gives 0.0 to 1.0 per column

    # ── Identify columns to drop ──────────────────────────
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()

    # ── Nothing to drop ───────────────────────────────────
    if not cols_to_drop:
        logger.info(
            f"No columns exceed {threshold * 100:.0f}% null threshold — "
            f"all {len(df.columns)} columns retained"
        )
        return df

    # ── Log what's being dropped ──────────────────────────
    logger.info(
        f"Dropping {len(cols_to_drop)} columns exceeding "
        f"{threshold * 100:.0f}% null threshold"
    )

  
    # ── Drop and report ───────────────────────────────────
    df_cleaned = df.drop(columns=cols_to_drop)

    logger.info(
        f"Shape before: {df.shape} | "
        f"Shape after:  {df_cleaned.shape} | "
        f"Columns removed: {len(cols_to_drop)}"
    )

    return df_cleaned

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    """
    Remove duplicate rows from the DataFrame.
    For fraud data, duplicated TransactionIDs are a data pipeline error
    and must be removed before any modeling.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with duplicate rows removed

    Raises:
        TypeError: If df is not a DataFrame
    """

    # ── Input validation ──────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")

    if df.empty:
        logger.warning("Received an empty DataFrame — skipping duplicate removal")
        return df

    # ── Check for full row duplicates ─────────────────────
    n_duplicates = df.duplicated().sum()

    if n_duplicates == 0:
        logger.info("No duplicate rows found — all rows retained")
        return df

    # ── Drop and report ───────────────────────────────────
    df_cleaned = df.drop_duplicates()

    logger.info(
        f"Dropped {n_duplicates:,} duplicate rows | "
        f"Shape before: {df.shape} | "
        f"Shape after: {df_cleaned.shape}"
    )

    return df_cleaned


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:

    """
    Attempt to convert object columns that are actually numeric.
    Pandas sometimes reads numeric columns as strings if a single
    row contains an unexpected character.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with corrected data types

    Raises:
        TypeError: If df is not a DataFrame
    """

    # ── Input validation ──────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")

    if df.empty:
        logger.warning("Received an empty DataFrame — skipping dtype fixing")
        return df

    # ── Attempt numeric conversion on object cols ─────────
    object_cols   = df.select_dtypes(include="object").columns.tolist()
    converted     = []
    not_converted = []

    for col in object_cols:
        converted_col = pd.to_numeric(df[col], errors="coerce")

        # Only convert if >90% of values successfully parsed as numeric
        # Avoids converting true categorical columns like card type
        success_rate = converted_col.notna().mean()

        if success_rate > 0.90:
            df[col] = converted_col
            converted.append(col)
        else:
            not_converted.append(col)

    # ── Report ────────────────────────────────────────────
    if converted:
        logger.info(
            f"Converted {len(converted)} object columns to numeric: "
            f"{converted[:5]}{'...' if len(converted) > 5 else ''}"
        )

    logger.info(
        f"Retained {len(not_converted)} true categorical columns as object"
    )
    logger.info(
        f"Final dtypes | "
        f"Numeric: {len(df.select_dtypes(include='number').columns)} | "
        f"Categorical: {len(df.select_dtypes(include='object').columns)}"
    )

    return df





def build_base() -> pd.DataFrame:

    """
    Full pipeline to build the base dataset:
        1. Load raw transaction and identity data
        2. Merge on TransactionID
        3. Remove columns with > NULL_THRESHOLD missing values
        4. Remove duplicate rows
        5. Fix data types
        6. Save to BASE_DATA path

    Returns:
        Base DataFrame ready for versioning
    """

    logger.info("=" * 50)
    logger.info("Starting base dataset build pipeline")
    logger.info("=" * 50)

    # ── Step 1: Load ──────────────────────────────────────
    logger.info("Step 1/6 — Loading raw data")
    df_transaction = load_data(RAW_TRANSACTION)
    df_identity    = load_data(RAW_IDENTITY)

    # ── Step 2: Merge ─────────────────────────────────────
    logger.info("Step 2/6 — Merging datasets")
    df = merge_data(df_transaction, df_identity)

    # ── Step 3: Remove high-null columns ──────────────────
    logger.info("Step 3/6 — Removing high-null columns")
    df = remove_high_null_columns(df, threshold=NULL_THRESHOLD)

    # ── Step 4: Remove duplicates ─────────────────────────
    logger.info("Step 4/6 — Removing duplicate rows")
    df = drop_duplicates(df)

    # ── Step 5: Fix data types ────────────────────────────
    logger.info("Step 5/6 — Fixing data types")
    df = fix_dtypes(df)

    # ── Step 6: Save ──────────────────────────────────────
    logger.info("Step 6/6 — Saving base dataset")
    save_data(df, BASE_DATA)

    logger.info("=" * 50)
    logger.info(f"Base dataset build complete | Final shape: {df.shape}")
    logger.info("=" * 50)

    return df


if __name__ == "__main__":
    # Run directly from terminal: python -m src.data_processing.base_builder
    df_base = build_base()
    print(df_base.head())