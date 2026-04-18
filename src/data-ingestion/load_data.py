"""
this file is used for data ingestion / loading.
- data/train/train_identity.csv
- data/train/train_transaction.csv
"""

import pandas as pd
import logging 
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(input_path: str | Path ) -> pd.DataFrame:
     """
    Load a CSV file and return a cleaned DataFrame.

    Args:
        input_path: Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the file is empty or not a valid CSV.
    """
     
     input_path = Path(input_path)
     
     # Check if file exists 
     if not input_path.exists():
          raise FileNotFoundError(f"Input file not found: {input_path}")
     
     # Check if format is csv
     if input_path.suffix.lower() != '.csv':
          raise ValueError(f"Expected a .csv but got {input_path.suffix} ")
     
     logger.info(f"Loading Data From: {input_path}")
     df = pd.read_csv(input_path)

     # check if df is not empty
     if df.empty:
          raise ValueError(f"File is Empty: {input_path}")
     
     logger.info(f"Loaded data with {len(df):,} rows and {len(df.columns)} columns")
     return df



     
     

          


