from pathlib import Path


# ── Project Root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # project root



# ── Data Paths ────────────────────────────────────────────
RAW_TRANSACTION  = ROOT / "data/raw/train_transaction.csv"
RAW_IDENTITY     = ROOT / "data/raw/train_identity.csv"
BASE_DATA        = ROOT / "data/base/train_base.csv"

V1_DATA = ROOT / "data/versions/v1_xgboost/train_v1.csv"
V2_DATA = ROOT / "data/versions/v2_iso/train_v2.csv"
V3_DATA = ROOT / "data/versions/v3_xgboost/train_v3.csv"
V4_DATA = ROOT / "data/versions/v4_gnn/train_v4.csv"

# ── Data Processing Metrics ────────────────────────────────────────────
NULL_THRESHOLD = 0.80   
