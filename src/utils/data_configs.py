from pathlib import Path

# ── Project Root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ── Raw Data Paths ────────────────────────────────────────
RAW_TRANSACTION = ROOT / "data/raw/train_transaction.csv"
RAW_IDENTITY    = ROOT / "data/raw/train_identity.csv"

# ── Base ────────────────────────────────────────
BASE_DATA  = ROOT / "data/base/base.csv"
BASE_TRAIN = ROOT / "data/base/train.csv"
BASE_VAL   = ROOT / "data/base/val.csv"
BASE_TEST  = ROOT / "data/base/test.csv"

# ── Split ratios ──────────────────────────────────────────
TRAIN_RATIO = 0.70   # first 70% of time → train
VAL_RATIO   = 0.15   # next 15% of time  → validation
TEST_RATIO  = 0.15   # last 15% of time  → test

# ── V1: XGBoost Baseline ──────────────────────────────────
V1_TRAIN = ROOT / "data/versions/v1_xgboost/train.csv"
V1_TEST  = ROOT / "data/versions/v1_xgboost/test.csv"

# ── V2: Isolation Forest ──────────────────────────────────
V2_TRAIN = ROOT / "data/versions/v2_iso/train.csv"
V2_TEST  = ROOT / "data/versions/v2_iso/test.csv"

# ── V3: XGBoost + Feature Engineering ────────────────────
V3_TRAIN = ROOT / "data/versions/v3_xgboost_fe/train.csv"
V3_TEST  = ROOT / "data/versions/v3_xgboost_fe/test.csv"

# ── V4: GNN ───────────────────────────────────────────────
V4_TRAIN = ROOT / "data/versions/v4_gnn/train.csv"
V4_TEST  = ROOT / "data/versions/v4_gnn/test.csv"

# ── Data Processing Settings ──────────────────────────────
NULL_THRESHOLD = 0.80
TARGET_COL     = "isFraud"
RANDOM_STATE   = 42