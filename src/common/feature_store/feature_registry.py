

# ── Time Features ──────────────────────────────────────────────
TIME_FEATURES = [
    "hour",
    "day_of_week",
    "days_since_start",
    "is_weekend",
    "is_high_risk_hour",
    "time_of_day",
]

# ── Amount Features ────────────────────────────────────────
AMOUNT_FEATURES = [
    "amt_log",           # log1p — fixes skew=17.64
    "amt_cents",         # decimal part: 0.00, 0.95, etc.
    "amt_cents_type",    # 0=round, 1=charm(.95), 2=other — 10x fraud rate difference
    "amt_is_very_low",   # 1 if amt < $26  (5.5% fraud)
    "amt_is_high",       # 1 if amt > $280 (5.1% fraud)
    "amt_bucket",        # decile 0-9, fitted on train only — no leakage
]

# ── Null Flag Features ─────────────────────────────────────
NULL_FLAG_FEATURES = [
    "id_data_present",   # 1 if device identity block present (corr=1.0 covers 19+ cols)
    "id_13_was_null",    # 1 if id_13 missing — independent null pattern, diff=0.292
    "id_16_was_null",    # 1 if id_16 missing — independent null pattern, diff=0.265
]

# ── Aggregation Features ───────────────────────────────────

AGGREGATION_FEATURES = [
    "card1_txn_count",    # times card appeared in train (U-shape: 1-txn and 100+ riskier)
    "card1_mean_amt",     # avg spend for this card in train (used in ratio features)
    "card1_std_amt",      # spend variation — low std = bot-like, 25th pct $41 vs $98
    "card1_is_high_freq", # 1 if card had 100+ txns in train (4.04% fraud rate)
   
    "email_fraud_rate",   # historical fraud rate for this domain — mail.com=19.8%
    "email_txn_count",    # how many txns this domain had in train
]


# ── Ratio Features ─────────────────────────────────────────
# Source: ratio_features.py | Type: Pure transform
# Depends on: aggregation_features must be applied first
# Smoothing: +1 on all denominators (SMOOTH constant)
# Capping: clipped at 500x to prevent outlier explosion
RATIO_FEATURES = [
    "amt_to_card1_mean_ratio",  # this txn vs card's avg spend — fraud deviated +$19.89 avg
    "amt_to_card1_std_ratio",   # std-deviation ratio — bot cards (std=0) get high ratio
]

# ── Graph Features ─────────────────────────────────────────
# Source: graph_features.py | Type: Fit + Apply (train only)
# EDA: device_txn_count 21–100 = 14.5% fraud (strongest signal in EDA)
#      card_email_pair bimodal: 186k pairs at 0%, 1,047 pairs at 100%

GRAPH_FEATURES = [
    "device_txn_count",           # device activity — 21–100 txns = 4x baseline fraud
    "card_device_is_high",        # 1 if card tied to 200+ devices — up to 16.6% fraud
    "addr_is_unique",             # 1 if address used by only 1 card — 0.0% fraud
    "card_email_pair_fraud_rate", # known bad card+email combo — bimodal 0% vs 100%
    "uid_fraud_rate",             # known bad card+addr combo  — 0.3% rows at 100%
]


# ══════════════════════════════════════════════════════════
#  MASTER LIST — import this in build_enriched.py
# ══════════════════════════════════════════════════════════
ALL_ENGINEERED_FEATURES = (
    TIME_FEATURES        # 6  features
    + AMOUNT_FEATURES    # 6  features
    + NULL_FLAG_FEATURES # 3  features
    + AGGREGATION_FEATURES # 6 features
    + RATIO_FEATURES     # 2  features
    + GRAPH_FEATURES     # 5  features
)
# Total: 28 engineered features
# These + SHAP top-50 raw features = final model input