

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
