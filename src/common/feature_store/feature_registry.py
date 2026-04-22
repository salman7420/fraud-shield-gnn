

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
    "amt_cents_type",    # 0=round, 1=charm(.95), 2=other — strongest amount signal
    "amt_is_very_low",   # 1 if amt < $26  (5.5% fraud rate)
    "amt_is_high",       # 1 if amt > $280 (5.1% fraud rate)
    "amt_bucket",        # decile 0-9, captures U-shaped fraud pattern
]