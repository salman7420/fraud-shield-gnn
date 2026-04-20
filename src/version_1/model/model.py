from src.utils.data_configs import V1_TRAIN, V1_VAL, V1_MODEL
from src.utils.logger import get_logger
from src.data_ingestion.load_data import load_data

import pandas as pd
import xgboost as xgb
from pathlib import Path


logger = get_logger(__name__) 

# ── Constants ──────────────────────────────────────────────────
TARGET_COL    = "isFraud"
V1_MODEL_PATH = V1_MODEL / "model.json"


# ──Separate features and target ─────────────────────
def separate_X_y(df: pd.DataFrame, target: str = TARGET_COL  ):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    logger.info(f"Features shape: {X.shape} | Target shape: {y.shape}")
    logger.info(f"Fraud rate: {y.mean():.4f} ({y.mean() * 100:.2f}%)")
    
    return X, y


# Save the trained model ────────────────────────────
def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)          # XGBoost native .json — stable across versions
    logger.info(f"Model saved to: {path}")


# Run for Pipeline ────────────────────────────
def run():
    logger.info("═══ V1 Model Process Started ═══════════════════════")

    # ── Load v1 data  ───────────────────────────────────────────
    v1_train = load_data(V1_TRAIN)
    v1_val = load_data(V1_VAL)

    # ── Step 2: Separate features and target ─────────────────────
    X_train, y_train = separate_X_y(v1_train, TARGET_COL)
    X_val,   y_val   = separate_X_y(v1_val, TARGET_COL)


    # ── Step 3: XGBoost config ─────────────────────────────────────
    # scale_pos_weight = non-fraud count / fraud count
    # = (1 - 0.0352) / 0.0352 ≈ 27.4
    # Tells XGBoost: "a fraud mistake costs 27x more than a non-fraud mistake"

    SCALE_POS_WEIGHT = (1 - y_train.mean()) / y_train.mean()
    logger.info(f"scale_pos_weight: {SCALE_POS_WEIGHT:.2f}")

    XGB_PARAMS = {
    "n_estimators":      1000,    # Max trees — early stopping will find optimal
    "learning_rate":     0.05,    # Small steps = more careful learning
    "max_depth":         6,       # How deep each tree grows
    "subsample":         0.8,     # Use 80% of rows per tree — reduces overfitting
    "colsample_bytree":  0.8,     # Use 80% of columns per tree
    "scale_pos_weight":  SCALE_POS_WEIGHT,  # Handle class imbalance
    "eval_metric":       "aucpr", # Optimize for AUPRC — best for imbalanced data
    "early_stopping_rounds": 50,  # Stop if no improvement for 50 rounds
    "random_state":      42,
    "n_jobs":            -1,      # Use all CPU cores
    "verbosity":         1
}

    # ── Step 4: Train ─────────────────────────────────────────────
    logger.info("Training XGBoost model...")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],   # Monitor val performance each round
        verbose=100                  # Print progress every 100 trees
    )

    best_iteration = model.best_iteration   
    logger.info(f"Training complete | Best iteration: {best_iteration}")
    
    # ── Step 5: Save the trained model ────────────────────────────
    save_model(model, V1_MODEL_PATH)

    # ── Step 6: Log training summary ──────────────────────────────
    logger.info("─── V1 XGBoost Training Summary ───────────────────")
    logger.info(f"  Best iteration      : {model.best_iteration}")
    logger.info(f"  Best val AUPRC      : {model.best_score:.4f}")
    logger.info(f"  Total trees built   : {model.best_iteration + 1}")
    logger.info(f"  Model saved to      : {V1_MODEL_PATH}")
    logger.info("───────────────────────────────────────────────────")
    logger.info("═══ V1 Model Process Completed ═════════════════════")
    
if __name__ == "__main__":
    run()


