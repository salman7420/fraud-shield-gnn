# src/common/feature_engineering/ratio_features.py
#
# Pure transform — no fitting needed.
# Depends on aggregation_features.py being applied first:
#   card1_mean_amt and card1_std_amt must exist in df.
#
# Call on train, val, and test independently with same logic.

from src.utils.logger import get_logger
import numpy as np
import pandas as pd

logger = get_logger(__name__)

# ── Smoothing constant ─────────────────────────────────────
# Added to denominators to prevent:
#   1. Division by zero  (mean_amt or std_amt = 0)
#   2. Exploding ratios  (very small denominators → huge ratios)
# Value = 1.0 chosen because TransactionAmt min = $0.25
# so +1 is small enough not to distort large values
# but large enough to stabilize near-zero denominators
SMOOTH = 1.0

# ── Ratio cap ──────────────────────────────────────────────
# Ratios can still be extreme even with smoothing
# e.g. a $5000 transaction for a card whose mean is $0.50
# Cap at 99th percentile equivalent → anything beyond is
# clipped to this value so model doesn't overfit to outliers
# Set conservatively high — real signal still preserved
RATIO_CAP = 500.0


def add_ratio_features(df: pd.DataFrame,
                       amt_col:      str = "TransactionAmt",
                       mean_col:     str = "card1_mean_amt",
                       std_col:      str = "card1_std_amt") -> pd.DataFrame:
    """
    Adds ratio features comparing this transaction's amount
    to the card's historical spending behavior.

    Requires aggregation_features.apply_aggregation_features()
    to have been called first on this dataframe.

    Features added:
      amt_to_card1_mean_ratio  — how much bigger/smaller this txn
                                 is vs the card's average spend
      amt_to_card1_std_ratio   — how many "steps" of variation
                                 this txn is away from card's normal

    Edge cases handled:
      - mean_amt or std_amt = 0  → SMOOTH (+1) prevents division by zero
      - missing prerequisite cols → warning + early return
      - extreme ratio values      → clipped at RATIO_CAP (500x)
      - NaN in amt/mean/std col  → ratio becomes NaN → XGBoost handles it
    """

    # ── Prerequisite check ─────────────────────────────────
    required = [amt_col, mean_col, std_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            f"Ratio features skipped — missing columns: {missing}. "
            f"Ensure apply_aggregation_features() ran first."
        )
        return df

    amt  = df[amt_col]
    mean = df[mean_col]
    std  = df[std_col]

    # ── amt_to_card1_mean_ratio ────────────────────────────
    # Question: "Is this transaction unusually large for this card?"
    #
    # Example A — legit:
    #   card mean = $55, this txn = $60 → ratio = 60/56 = 1.07  (normal)
    #
    # Example B — fraud:
    #   card mean = $55, this txn = $800 → ratio = 800/56 = 14.3 (alarming)
    #
    # Example C — new card (fallback mean = global median ~$68.95):
    #   this txn = $500 → ratio = 500/69.95 = 7.14 (suspicious)
    df["amt_to_card1_mean_ratio"] = np.clip(
        amt / (mean + SMOOTH),
        a_min = 0,
        a_max = RATIO_CAP
    )

    # ── amt_to_card1_std_ratio ─────────────────────────────
    # Question: "How far is this txn from the card's normal range?"
    #
    # Example A — consistent card (bot-like, std=$0):
    #   std = $0, this txn = $50 → ratio = 50/(0+1) = 50  (very high)
    #   → Bot cards with no variation history look alarming for any txn
    #
    # Example B — varied card (normal human, std=$150):
    #   this txn = $200 → ratio = 200/151 = 1.32  (barely unusual)
    #
    # Example C — card with large std ($500):
    #   this txn = $1000 → ratio = 1000/501 = 2.0  (within normal range)
    #
    # Note: std=0 cards (single-txn, filled by aggregation_features)
    #       will always get a high ratio → that's intentional and correct.
    #       A card with no variation history IS unusual for any transaction.
    df["amt_to_card1_std_ratio"] = np.clip(
        amt / (std + SMOOTH),
        a_min = 0,
        a_max = RATIO_CAP
    )

    logger.info(
        f"Ratio features added | "
        f"mean_ratio — min: {df['amt_to_card1_mean_ratio'].min():.2f} "
        f"max: {df['amt_to_card1_mean_ratio'].max():.2f} | "
        f"std_ratio  — min: {df['amt_to_card1_std_ratio'].min():.2f} "
        f"max: {df['amt_to_card1_std_ratio'].max():.2f}"
    )

    return df


if __name__ == "__main__":
    # Smoke test — covers all edge cases

    test_df = pd.DataFrame({
        "TransactionAmt": [60.0,   800.0,  50.0,   500.0,  0.25],
        "card1_mean_amt": [55.0,   55.0,   0.0,    68.95,  2000.0],
        "card1_std_amt":  [5.0,    5.0,    0.0,    0.0,    800.0],
        # Case 1: normal legit txn
        # Case 2: fraud — way above card mean
        # Case 3: zero mean (all $0 history) — SMOOTH prevents div/0
        # Case 4: new card (fallback values) — no std history
        # Case 5: tiny txn on high-value card — ratio near 0
    })

    result = add_ratio_features(test_df)
    print(result[[
        "TransactionAmt", "card1_mean_amt", "card1_std_amt",
        "amt_to_card1_mean_ratio", "amt_to_card1_std_ratio"
    ]])

    # Expected output:
    # Case 1: mean_ratio=60/56=1.07    std_ratio=60/6=10.0
    # Case 2: mean_ratio=800/56=14.3   std_ratio=800/6=133.3
    # Case 3: mean_ratio=50/1=50.0     std_ratio=50/1=50.0
    # Case 4: mean_ratio=500/69.95=7.1 std_ratio=500/1=500 (capped)
    # Case 5: mean_ratio=0.25/2001=0.0 std_ratio=0.25/801=0.0