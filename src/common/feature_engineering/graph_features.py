# src/common/feature_engineering/graph_features.py
#
# Graph-based features capturing entity relationships.
# All target-encoded features (fraud rates) MUST be fitted
# on train only — applying train's values to val/test.
#
# EDA Summary:
#   device_txn_count:           21-100 txns → 14.5% fraud (4x baseline) ← strongest signal
#   card_device_is_high:        degree > 200 → up to 16.6% fraud
#   addr_is_unique:             degree = 1  → 0.0% fraud (clean signal)
#   card_email_pair_fraud_rate: bimodal 0% vs 100% — known bad actor lookup
#   uid_fraud_rate:             0.3% rows at 100% fraud — card+addr blacklist

from src.utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)

# ── EDA-confirmed thresholds ───────────────────────────────
CARD_DEVICE_HIGH_THRESHOLD = 200   # degree > 200 → up to 16.6% fraud
GLOBAL_FRAUD_RATE          = 0.035 # fallback for unseen pairs in val/test


# ══════════════════════════════════════════════════════════
#  BLOCK 1 — DEVICE FEATURES
#  device_txn_count:    how many transactions this device has in train
#  card_device_is_high: flag for cards tied to 200+ unique devices
#
#  EDA findings:
#    6-20   device txns → 12.8% fraud
#    21-100 device txns → 14.5% fraud (strongest signal in entire EDA)
#    100+   device txns → 5.6%  fraud (drops — established devices)
#    card_device_degree > 200 → up to 16.6% fraud
# ══════════════════════════════════════════════════════════

def _fit_device_features(train_df: pd.DataFrame,
                         device_col: str = "DeviceInfo",
                         card_col:   str = "card1") -> dict:
    """
    Learns:
      device_txn_count_lookup  — {device: txn_count}
      card_device_degree_lookup — {card1: unique_device_count}

    Edge cases:
      - DeviceInfo null (78% of rows) → NaN in output, XGBoost handles natively
      - Device in val/test not in train → fallback = 1 (first time seen)
      - Card in val/test not in train  → fallback = 0 (no device history)
    """
    params = {}

    # device_txn_count — how many times each device appears in train
    if device_col in train_df.columns:
        device_counts = (
            train_df[device_col]
            .value_counts()
            .to_dict()
        )
        params["device_txn_count_lookup"] = device_counts
        params["device_txn_count_fallback"] = 1  # unseen device = first time

        logger.info(
            f"device_txn_count fitted — "
            f"{len(device_counts):,} unique devices"
        )

    # card_device_degree — unique devices per card1
    if card_col in train_df.columns and device_col in train_df.columns:
        card_device_degree = (
            train_df.groupby(card_col)[device_col]
            .nunique()
            .to_dict()
        )
        params["card_device_degree_lookup"]  = card_device_degree
        params["card_device_degree_fallback"] = 0  # unseen card = no device history

        logger.info(
            f"card_device_degree fitted — "
            f"{len(card_device_degree):,} unique cards"
        )

    return params


def _apply_device_features(df: pd.DataFrame,
                            params: dict,
                            device_col: str = "DeviceInfo",
                            card_col:   str = "card1") -> pd.DataFrame:
    """
    Applies device features to any split using fitted params.

    Features added:
      device_txn_count    — times this device appeared in train
                            NaN if DeviceInfo is null (78% of rows)
      card_device_is_high — 1 if card tied to 200+ unique devices in train
    """
    # device_txn_count
    if "device_txn_count_lookup" in params and device_col in df.columns:
        lookup   = params["device_txn_count_lookup"]
        fallback = params["device_txn_count_fallback"]

        df["device_txn_count"] = df[device_col].map(
            lambda d: lookup.get(d, fallback) if pd.notna(d) else np.nan
        )

        logger.info(
            f"device_txn_count applied | "
            f"null (no DeviceInfo): "
            f"{df['device_txn_count'].isna().sum():,} rows"
        )

    # card_device_is_high — derived from card_device_degree
    if "card_device_degree_lookup" in params and card_col in df.columns:
        lookup   = params["card_device_degree_lookup"]
        fallback = params["card_device_degree_fallback"]

        df["card_device_is_high"] = df[card_col].map(
            lambda c: int(lookup.get(c, fallback) > CARD_DEVICE_HIGH_THRESHOLD)
        )

        high_count = df["card_device_is_high"].sum()
        logger.info(
            f"card_device_is_high applied | "
            f"high-degree cards: {high_count:,} "
            f"({high_count/len(df)*100:.1f}% of rows)"
        )

    return df


# ══════════════════════════════════════════════════════════
#  BLOCK 2 — ADDRESS FEATURES
#  addr_is_unique: flag for addresses tied to only 1 card
#
#  EDA findings:
#    addr_card_degree = 1 → 0.0% fraud (193 rows, perfectly clean)
#    Very specific address never reused = traceable = legit signal
# ══════════════════════════════════════════════════════════

def _fit_addr_features(train_df: pd.DataFrame,
                       addr_col: str = "addr1",
                       card_col: str = "card1") -> dict:
    """
    Learns addr_card_degree for each address in train.

    Edge cases:
      - addr1 null (11.5% of rows) → NaN in output
      - Address in val/test not in train → fallback = 0 (unknown)
    """
    params = {}

    if addr_col in train_df.columns and card_col in train_df.columns:
        addr_card_degree = (
            train_df.groupby(addr_col)[card_col]
            .nunique()
            .to_dict()
        )
        params["addr_card_degree_lookup"]  = addr_card_degree
        params["addr_card_degree_fallback"] = 0  # unseen address = unknown

        unique_addrs = sum(1 for v in addr_card_degree.values() if v == 1)
        logger.info(
            f"addr_is_unique fitted — "
            f"{len(addr_card_degree):,} unique addresses | "
            f"degree=1 (unique): {unique_addrs:,}"
        )

    return params


def _apply_addr_features(df: pd.DataFrame,
                         params: dict,
                         addr_col: str = "addr1") -> pd.DataFrame:
    """
    Features added:
      addr_is_unique — 1 if this address was tied to exactly 1 card in train
                       0.0% fraud rate for these rows (EDA confirmed)
    """
    if "addr_card_degree_lookup" in params and addr_col in df.columns:
        lookup   = params["addr_card_degree_lookup"]
        fallback = params["addr_card_degree_fallback"]

        df["addr_is_unique"] = df[addr_col].map(
            lambda a: int(lookup.get(a, fallback) == 1)
            if pd.notna(a) else np.nan
        )

        logger.info(
            f"addr_is_unique applied | "
            f"unique addr rows: {df['addr_is_unique'].sum():,}"
        )

    return df


# ══════════════════════════════════════════════════════════
#  BLOCK 3 — PAIR FRAUD RATES (TARGET ENCODING)
#  card_email_pair_fraud_rate: fraud rate for card+email combo
#  uid_fraud_rate:             fraud rate for card+addr combo
#
#  EDA findings:
#    card_email_pair: 186,380 pairs at 0%, 1,047 pairs at 100%
#    uid_fraud_rate:  48.6% rows at 0%, 0.3% rows at 100%
#    Both bimodal — strong known-bad-actor lookup signal
#
#  ⚠️ TARGET ENCODING — uses isFraud to compute rates
#     MUST be fitted on train only. Applying to val/test
#     uses train's rates — no leakage.
# ══════════════════════════════════════════════════════════

def _fit_pair_fraud_rates(train_df:   pd.DataFrame,
                          target_col: str = "isFraud",
                          card_col:   str = "card1",
                          email_col:  str = "P_emaildomain",
                          addr_col:   str = "addr1") -> dict:
    """
    Learns fraud rates for:
      card1 + P_emaildomain combination
      card1 + addr1 combination (uid)

    Edge cases:
      - Pair not seen in val/test → fallback = global fraud rate (0.035)
      - Null in email or addr     → pair key includes "nan" string
                                    → mapped to fallback
    """
    params = {}
    global_rate = float(train_df[target_col].mean())
    params["pair_global_fraud_rate"] = global_rate

    # card + email pair fraud rate
    if card_col in train_df.columns and email_col in train_df.columns:
        train_df["_card_email_pair"] = (
            train_df[card_col].astype(str) + "_" +
            train_df[email_col].astype(str)
        )
        card_email_rates = (
            train_df.groupby("_card_email_pair")[target_col]
            .mean()
            .to_dict()
        )
        train_df.drop(columns=["_card_email_pair"], inplace=True)

        params["card_email_pair_rates"] = card_email_rates

        pairs_at_100 = sum(1 for v in card_email_rates.values() if v == 1.0)
        pairs_at_0   = sum(1 for v in card_email_rates.values() if v == 0.0)
        logger.info(
            f"card_email_pair_fraud_rate fitted — "
            f"{len(card_email_rates):,} pairs | "
            f"100% fraud: {pairs_at_100:,} | "
            f"0% fraud: {pairs_at_0:,}"
        )

    # uid fraud rate (card + addr)
    if card_col in train_df.columns and addr_col in train_df.columns:
        train_df["_uid"] = (
            train_df[card_col].astype(str) + "_" +
            train_df[addr_col].astype(str)
        )
        uid_rates = (
            train_df.groupby("_uid")[target_col]
            .mean()
            .to_dict()
        )
        train_df.drop(columns=["_uid"], inplace=True)

        params["uid_fraud_rates"] = uid_rates

        uids_at_100 = sum(1 for v in uid_rates.values() if v == 1.0)
        uids_at_0   = sum(1 for v in uid_rates.values() if v == 0.0)
        logger.info(
            f"uid_fraud_rate fitted — "
            f"{len(uid_rates):,} uid pairs | "
            f"100% fraud: {uids_at_100:,} | "
            f"0% fraud: {uids_at_0:,}"
        )

    return params


def _apply_pair_fraud_rates(df:        pd.DataFrame,
                             params:    dict,
                             card_col:  str = "card1",
                             email_col: str = "P_emaildomain",
                             addr_col:  str = "addr1") -> pd.DataFrame:
    """
    Features added:
      card_email_pair_fraud_rate — historical fraud rate for card+email combo
                                   unseen pairs → global fraud rate (0.035)
      uid_fraud_rate             — historical fraud rate for card+addr combo
                                   unseen pairs → global fraud rate (0.035)
    """
    fallback = params.get("pair_global_fraud_rate", GLOBAL_FRAUD_RATE)

    # card_email_pair_fraud_rate
    if "card_email_pair_rates" in params:
        lookup = params["card_email_pair_rates"]
        df["card_email_pair_fraud_rate"] = (
            df[card_col].astype(str) + "_" + df[email_col].astype(str)
        ).map(lambda p: lookup.get(p, fallback))

        logger.info(
            f"card_email_pair_fraud_rate applied | "
            f"unseen pairs (fallback): "
            f"{df['card_email_pair_fraud_rate'].eq(fallback).sum():,}"
        )

    # uid_fraud_rate
    if "uid_fraud_rates" in params:
        lookup = params["uid_fraud_rates"]
        df["uid_fraud_rate"] = (
            df[card_col].astype(str) + "_" + df[addr_col].astype(str)
        ).map(lambda p: lookup.get(p, fallback))

        logger.info(
            f"uid_fraud_rate applied | "
            f"unseen uids (fallback): "
            f"{df['uid_fraud_rate'].eq(fallback).sum():,}"
        )

    return df


# ══════════════════════════════════════════════════════════
#  MASTER FIT / APPLY — called by build_enriched.py
# ══════════════════════════════════════════════════════════

def fit_graph_features(train_df: pd.DataFrame) -> dict:
    """
    Master fit — call on train ONLY.
    Runs all graph feature fits and merges into one params dict.
    """
    params = {}
    params.update(_fit_device_features(train_df))
    params.update(_fit_addr_features(train_df))
    params.update(_fit_pair_fraud_rates(train_df))

    logger.info("All graph features fitted ✓")
    return params


def apply_graph_features(df: pd.DataFrame,
                          params: dict) -> pd.DataFrame:
    """
    Master apply — call on train, val, AND test with SAME params.
    Applies all graph features in correct dependency order.
    """
    df = _apply_device_features(df, params)
    df = _apply_addr_features(df, params)
    df = _apply_pair_fraud_rates(df, params)

    logger.info("All graph features applied ✓")
    return df


# ══════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_sample = pd.DataFrame({
        "card1":          [101, 101, 102, 103, 103, 101],
        "DeviceInfo":     ["Win/Chrome", "Win/Chrome", "iOS/Safari",
                           "Android",    "Android",    "Win/Firefox"],
        "addr1":          [10.0, 10.0, 20.0, 30.0, 30.0, 10.0],
        "P_emaildomain":  ["gmail.com", "mail.com", "gmail.com",
                           "mail.com",  "mail.com",  "gmail.com"],
        "isFraud":        [0, 1, 0, 1, 1, 0],
    })

    val_sample = pd.DataFrame({
        "card1":          [101, 999,        102],
        "DeviceInfo":     ["Win/Chrome", "NewDevice", "iOS/Safari"],
        "addr1":          [10.0, 99.0,    20.0],
        "P_emaildomain":  ["gmail.com", "unknown.com", "gmail.com"],
    })

    params = fit_graph_features(train_sample)
    result = apply_graph_features(val_sample, params)

    print(result[[
        "card1", "device_txn_count", "card_device_is_high",
        "addr_is_unique", "card_email_pair_fraud_rate", "uid_fraud_rate"
    ]])

    # Expected:
    # card1=101, Win/Chrome → device_txn_count=2, card_device_is_high=0
    # card1=999, NewDevice  → device_txn_count=1(fallback), card_device_is_high=0
    # card1=102, iOS/Safari → device_txn_count=1
    # addr=10  → addr_is_unique=0 (card 101 used it — degree > 1)
    # addr=99  → addr_is_unique=0 (unseen → fallback=0)
    # addr=20  → addr_is_unique=1 (only card 102 used it)
    # gmail+101 → fraud_rate=0.0  (1 txn, 0 fraud)
    # unknown   → fraud_rate=0.333(global fallback)