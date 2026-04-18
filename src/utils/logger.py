# src/utils/logger.py
import logging
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for any module.
    Call with get_logger(__name__) from any file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    return logging.getLogger(name)