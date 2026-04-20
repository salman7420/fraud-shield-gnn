# ROOT/src/version_1/pipeline/pipeline.py

from src.utils.logger import get_logger
import src.version_1.preprocessing.processing as preprocessor
import src.version_1.model.model as model
import src.version_1.evaluation.evaluator as evaluator

import time

logger = get_logger(__name__)


def run():
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║        V1 XGBoost Pipeline Started               ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    pipeline_start = time.time()

    # ── Stage 1: Preprocessing ────────────────────────────────
    logger.info("── Stage 1/3: Preprocessing ────────────────────────")
    stage_start = time.time()
    preprocessor.run()
    logger.info(f"── Stage 1 completed in {time.time() - stage_start:.1f}s ──────────")

    # ── Stage 2: Model Training ───────────────────────────────
    logger.info("── Stage 2/3: Model Training ───────────────────────")
    stage_start = time.time()
    model.run()
    logger.info(f"── Stage 2 completed in {time.time() - stage_start:.1f}s ──────────")

    # ── Stage 3: Evaluation ───────────────────────────────────
    logger.info("── Stage 3/3: Evaluation ───────────────────────────")
    stage_start = time.time()
    evaluator.run()
    logger.info(f"── Stage 3 completed in {time.time() - stage_start:.1f}s ──────────")

    # ── Pipeline Summary ──────────────────────────────────────
    total_time = time.time() - pipeline_start
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║        V1 XGBoost Pipeline Completed             ║")
    logger.info(f"║        Total time: {total_time:.1f}s{' ' * (30 - len(f'{total_time:.1f}'))}║")
    logger.info("╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    run()