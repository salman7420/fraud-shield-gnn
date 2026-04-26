# src/version_2/v2_pipeline.py
#
# V2 Orchestration Pipeline — CatBoost
# Runs the full V2 pipeline in one command:
#   Stage 2B → catboost_prep.py
#   Stage 3  → train_catboost.py
#   Stage 4  → evaluate_catboost.py
#
# Usage (from project ROOT):
#   python -m src.version_2.pipeline.v2_pipeline

import sys
import time
import traceback
from pathlib import Path

# ── Project root on path ───────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.version_2.preprocessing.catboost_prep  import run_catboost_prep
from src.version_2.model.train         import run_training
from src.version_2.evaluation.evaluate_catboost      import run_evaluation

logger = get_logger(__name__)


def main() -> None:
    total_start = time.time()

    logger.info("=" * 60)
    logger.info("V2 CATBOOST PIPELINE — START")
    logger.info("=" * 60)

    # ── Stage timings tracker ──────────────────────────────
    timings = {}

    # ══════════════════════════════════════════════════════
    #  STAGE 2B — DATA PREP
    # ══════════════════════════════════════════════════════
    try:
        logger.info("\n" + "─" * 60)
        logger.info("STAGE 2B — CatBoost Data Prep")
        logger.info("─" * 60)
        t = time.time()

        run_catboost_prep(mode="all")

        timings["catboost_prep"] = round(time.time() - t, 1)
        logger.info(
            f"Stage 2B complete ✓  ({timings['catboost_prep']}s)"
        )

    except Exception as e:
        logger.error(f"Stage 2B FAILED: {e}")
        logger.error(traceback.format_exc())
        logger.error("Pipeline aborted — fix catboost_prep.py and re-run")
        sys.exit(1)

    # ══════════════════════════════════════════════════════
    #  STAGE 3 — TRAINING
    # ══════════════════════════════════════════════════════
    try:
        logger.info("\n" + "─" * 60)
        logger.info("STAGE 3 — CatBoost Training + Hyperparameter Tuning")
        logger.info("─" * 60)
        t = time.time()

        metrics = run_training()

        timings["training"] = round(time.time() - t, 1)
        logger.info(
            f"Stage 3 complete ✓  ({timings['training']}s)"
        )

    except Exception as e:
        logger.error(f"Stage 3 FAILED: {e}")
        logger.error(traceback.format_exc())
        logger.error("Pipeline aborted — fix train_catboost.py and re-run")
        sys.exit(1)

    # ══════════════════════════════════════════════════════
    #  STAGE 4 — EVALUATION
    # ══════════════════════════════════════════════════════
    try:
        logger.info("\n" + "─" * 60)
        logger.info("STAGE 4 — Evaluation + Plots")
        logger.info("─" * 60)
        t = time.time()

        eval_metrics = run_evaluation()

        timings["evaluation"] = round(time.time() - t, 1)
        logger.info(
            f"Stage 4 complete ✓  ({timings['evaluation']}s)"
        )

    except Exception as e:
        logger.error(f"Stage 4 FAILED: {e}")
        logger.error(traceback.format_exc())
        logger.error(
            "Training succeeded but evaluation failed.\n"
            "Model is saved — run evaluate_catboost.py separately."
        )
        sys.exit(1)

    # ══════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════
    total_elapsed = round(time.time() - total_start, 1)

    logger.info("\n" + "=" * 60)
    logger.info("V2 CATBOOST PIPELINE — COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Stage 2B  (prep)      : {timings['catboost_prep']}s")
    logger.info(f"  Stage 3   (training)  : {timings['training']}s")
    logger.info(f"  Stage 4   (evaluation): {timings['evaluation']}s")
    logger.info(f"  Total                 : {total_elapsed}s")
    logger.info("─" * 60)
    logger.info("  FINAL METRICS (val set):")
    logger.info(f"  AUC-ROC  : {eval_metrics['auc_roc']:.5f}")
    logger.info(f"  AUPRC    : {eval_metrics['auprc']:.5f}  ← primary")
    logger.info(f"  F1       : {eval_metrics['f1']:.5f}")
    logger.info(f"  Precision: {eval_metrics['precision']:.5f}")
    logger.info(f"  Recall   : {eval_metrics['recall']:.5f}")
    logger.info("─" * 60)
    logger.info("  OUTPUTS SAVED:")
    logger.info("  data/versions/v2_catboost/   ← prep splits")
    logger.info("  artifacts/catboost_cat_features.json")
    logger.info("  src/version_2/result/model.cbm")
    logger.info("  src/version_2/result/eval_report.json")
    logger.info("  src/version_2/result/plots/  ← 5 plots")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()