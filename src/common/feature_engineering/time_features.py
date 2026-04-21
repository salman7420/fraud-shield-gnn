# src/common/feature_engineering/time_features.py

from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

# Hours 5–10 have 2–3x baseline fraud rate (confirmed by EDA)
HIGH_RISK_HOURS = set(range(5, 11))


def add_time_features(df: pd.DataFrame,
                      dt_col: str = "TransactionDT") -> pd.DataFrame:
    """
    Extracts time-based features from TransactionDT (seconds offset).
    Pure transform — no fitting needed.
    Call on train, val, and test independently.
    """
    if dt_col not in df.columns:
        logger.warning(f"'{dt_col}' not found — skipping time features")
        return df

    dt = df[dt_col]

    # ── Core time features ─────────────────────────────────
    df["hour"]             = (dt // 3600) % 24
    df["day_of_week"]      = (dt // 86400) % 7
    df["days_since_start"] = dt // 86400

    # ── Derived flags (from EDA) ───────────────────────────
    df["is_weekend"]       = (df["day_of_week"] >= 5).astype(int)
    df["is_high_risk_hour"] = df["hour"].isin(HIGH_RISK_HOURS).astype(int)

    # ── Time of day bins ───────────────────────────────────
    # night=0-4, morning=5-11, afternoon=12-16, evening=17-23
    df["time_of_day"] = pd.cut(
        df["hour"],
        bins    = [-1, 4, 11, 16, 23],
        labels  = [0, 1, 2, 3]         # 0=night, 1=morning, 2=afternoon, 3=evening
    ).astype(int)

    logger.info(
        f"Time features added: hour, day_of_week, days_since_start, "
        f"is_weekend, is_high_risk_hour, time_of_day"
    )

    logger.info("Time features added ✓")
    
    return df


if __name__ == "__main__":
    # Quick smoke test
    test_df = pd.DataFrame({
        "TransactionDT": [86400, 3600*8, 3600*14, 3600*20],
        "isFraud":       [0, 1, 0, 0]
    })
    result = add_time_features(test_df)
    print(result[["TransactionDT", "hour", "day_of_week",
                  "is_high_risk_hour", "time_of_day"]])