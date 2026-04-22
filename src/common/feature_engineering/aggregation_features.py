# src/common/feature_engineering/aggregation_features.py

from src.utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)

# ── EDA-confirmed thresholds ───────────────────────────────
# Cell 2: cards with 100+ transactions → 4.04% fraud (highest bucket)
HIGH_FREQ_THRESHOLD = 100


# ══════════════════════════════════════════════════════════
#  CARD1 BLOCK
#  EDA findings:
#    - 12,242 unique cards, fraud rate 0%–100% per card
#    - U-shaped fraud by count: 1-txn (3.49%) and 100+ (4.04%) riskiest
#    - Fraud txns deviate +$19.89 above card mean vs -$0.72 for legit
#    - Fraud cards 25th pct std = $41 vs $98 for legit (bot-like consistency)
# ══════════════════════════════════════════════════════════

def fit_card1_features(train_df: pd.DataFrame,
                       amt_col: str = "TransactionAmt",
                       key_col: str = "card1") -> dict:
    """
    STEP 1 — Call on train ONLY.
    Learns per-card stats: txn_count, mean_amt, std_amt, is_high_freq.

    Edge cases:
      - std_amt NaN  (card seen once)    → stored as 0.0
      - unseen cards in val/test         → fallback values in params
      - mean_amt = 0 (all $0 txns)      → handled in ratio_features with +1
    """
    if key_col not in train_df.columns:
        logger.warning(f"'{key_col}' not found — skipping card1 fit")
        return {}

    card1_stats = (
        train_df.groupby(key_col)[amt_col]
        .agg(
            card1_txn_count="count",
            card1_mean_amt="mean",
            card1_std_amt="std",
        )
        .reset_index()
    )

    # std = NaN for single-txn cards → fill with 0.0 (no variation history)
    card1_stats["card1_std_amt"] = card1_stats["card1_std_amt"].fillna(0.0)

    # High frequency flag
    card1_stats["card1_is_high_freq"] = (
        card1_stats["card1_txn_count"] >= HIGH_FREQ_THRESHOLD
    ).astype(int)

    card1_lookup = card1_stats.set_index(key_col).to_dict(orient="index")

    # Fallback for cards never seen in train
    fallback = {
        "card1_txn_count":    1,
        "card1_mean_amt":     float(train_df[amt_col].median()),
        "card1_std_amt":      0.0,
        "card1_is_high_freq": 0,
    }

    params = {
        "card1_lookup":   card1_lookup,
        "card1_fallback": fallback,
    }

    logger.info(
        f"card1 fitted — {len(card1_lookup):,} unique cards | "
        f"high-freq (>={HIGH_FREQ_THRESHOLD}): "
        f"{card1_stats['card1_is_high_freq'].sum():,}"
    )
    return params


def apply_card1_features(df: pd.DataFrame,
                         params: dict,
                         key_col: str = "card1") -> pd.DataFrame:
    """
    STEP 2 — Call on train, val, AND test with SAME params.

    Features added:
      card1_txn_count    — times this card appeared in train
      card1_mean_amt     — average spend for this card in train
      card1_std_amt      — spend variation for this card in train
      card1_is_high_freq — 1 if card had 100+ txns in train
    """
    if key_col not in df.columns:
        logger.warning(f"'{key_col}' not found — skipping card1 apply")
        return df

    lookup   = params.get("card1_lookup",   {})
    fallback = params.get("card1_fallback", {})

    def get_stat(card_val, stat):
        return lookup.get(card_val, fallback).get(stat, fallback[stat])

    df["card1_txn_count"]    = df[key_col].map(lambda c: get_stat(c, "card1_txn_count"))
    df["card1_mean_amt"]     = df[key_col].map(lambda c: get_stat(c, "card1_mean_amt"))
    df["card1_std_amt"]      = df[key_col].map(lambda c: get_stat(c, "card1_std_amt"))
    df["card1_is_high_freq"] = df[key_col].map(lambda c: get_stat(c, "card1_is_high_freq"))

    unseen = df[key_col].map(lambda c: c not in lookup).sum()
    logger.info(f"card1 applied | unseen cards in this split: {unseen:,}")
    return df


# ══════════════════════════════════════════════════════════
#  EMAIL BLOCK
#  EDA findings:
#    - 59 unique domains, fraud rate 0%–19.8% per domain
#    - mail.com=19.8%, aim.com=16.7% vs gmx.de=0%, netzero=0%
#    - 6x difference between riskiest and safest domains
#    - ISP-assigned emails (netzero, windstream) = 0% fraud
#    - email_fraud_rate is TARGET ENCODED → must fit on train only
# ══════════════════════════════════════════════════════════

def fit_email_features(train_df: pd.DataFrame,
                       target_col: str = "isFraud",
                       key_col:    str = "P_emaildomain") -> dict:
    """
    STEP 1 — Call on train ONLY.
    Learns per-domain fraud rate and transaction count.

    Edge cases:
      - Domain not seen in val/test  → fallback = global fraud rate
      - Domain null in train         → excluded from lookup
      - Domains with very few txns   → their rates are noisy but kept
    """
    if key_col not in train_df.columns:
        logger.warning(f"'{key_col}' not found — skipping email fit")
        return {}

    email_stats = (
        train_df.groupby(key_col)[target_col]
        .agg(
            email_fraud_rate="mean",
            email_txn_count="count",
        )
        .reset_index()
    )

    email_lookup = email_stats.set_index(key_col).to_dict(orient="index")

    # Fallback for domains never seen in train (new domain in val/test)
    # Use global fraud rate — "unknown domain = average risk"
    global_fraud_rate = float(train_df[target_col].mean())

    fallback = {
        "email_fraud_rate": global_fraud_rate,
        "email_txn_count":  1,
    }

    params = {
        "email_lookup":       email_lookup,
        "email_fallback":     fallback,
        "global_fraud_rate":  global_fraud_rate,
    }

    logger.info(
        f"email fitted — {len(email_lookup):,} unique domains | "
        f"global fraud rate (fallback): {global_fraud_rate:.4f}"
    )
    return params


def apply_email_features(df: pd.DataFrame,
                         params: dict,
                         key_col: str = "P_emaildomain") -> pd.DataFrame:
    """
    STEP 2 — Call on train, val, AND test with SAME params.

    Features added:
      email_fraud_rate — historical fraud rate for this email domain
      email_txn_count  — how many transactions this domain had in train
    """
    if key_col not in df.columns:
        logger.warning(f"'{key_col}' not found — skipping email apply")
        return df

    lookup   = params.get("email_lookup",   {})
    fallback = params.get("email_fallback", {})

    def get_stat(domain, stat):
        return lookup.get(domain, fallback).get(stat, fallback[stat])

    df["email_fraud_rate"] = df[key_col].map(lambda d: get_stat(d, "email_fraud_rate"))
    df["email_txn_count"]  = df[key_col].map(lambda d: get_stat(d, "email_txn_count"))

    unseen = df[key_col].map(lambda d: d not in lookup).sum()
    logger.info(f"email applied | unseen domains in this split: {unseen:,}")
    return df


# ══════════════════════════════════════════════════════════
#  COMBINED FIT / APPLY — called by build_enriched.py
# ══════════════════════════════════════════════════════════

def fit_aggregation_features(train_df: pd.DataFrame) -> dict:
    """
    Master fit — call on train ONLY.
    Runs all aggregation fits and merges into one params dict.
    This is the only function build_enriched.py needs to call for fitting.
    """
    params = {}
    params.update(fit_card1_features(train_df))
    params.update(fit_email_features(train_df))

    logger.info("All aggregation features fitted ✓")
    return params


def apply_aggregation_features(df: pd.DataFrame,
                                params: dict) -> pd.DataFrame:
    """
    Master apply — call on train, val, AND test with SAME params.
    Runs all aggregation applies in the correct order.
    This is the only function build_enriched.py needs to call for applying.
    """
    df = apply_card1_features(df, params)
    df = apply_email_features(df, params)

    logger.info("All aggregation features applied ✓")
    return df


# ══════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_sample = pd.DataFrame({
        "card1":          [101, 101, 101, 102, 103, 103],
        "TransactionAmt": [50.0, 60.0, 55.0, 200.0, 10.0, 10.0],
        "P_emaildomain":  ["gmail.com", "mail.com", "gmail.com",
                           "yahoo.com", "gmail.com", "mail.com"],
        "isFraud":        [0, 1, 0, 0, 1, 1],
    })

    val_sample = pd.DataFrame({
        "card1":          [101, 102, 999],          # 999 = unseen card
        "TransactionAmt": [500.0, 50.0, 80.0],
        "P_emaildomain":  ["gmail.com", "new-domain.com", "mail.com"],
        # new-domain.com = unseen domain → should get global fraud rate
    })

    params = fit_aggregation_features(train_sample)
    result = apply_aggregation_features(val_sample, params)

    print(result[[
        "card1", "card1_txn_count", "card1_mean_amt",
        "card1_std_amt", "card1_is_high_freq",
        "P_emaildomain", "email_fraud_rate", "email_txn_count"
    ]])

    # Expected:
    # card1=101   → count=3, mean=$55,  std≈$5.0,  high_freq=0
    # card1=102   → count=1, mean=$200, std=$0.0,  high_freq=0
    # card1=999   → count=1, mean=$55(median fallback), std=0, high_freq=0
    # gmail.com   → fraud_rate=0.333, count=3
    # new-domain  → fraud_rate=0.5(global), count=1
    # mail.com    → fraud_rate=1.0,   count=2