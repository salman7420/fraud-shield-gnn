"""
this file is used for data ingestion / loading.
- raw/train/train_identity.csv
- raw/train/train_transaction.csv
"""

import pandas as pd
from src.utils.logger import get_logger
from pathlib import Path

logger = get_logger(__name__) 

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



def save_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df:          DataFrame to save.
        output_path: Path to save the output CSV file.

    Raises:
        ValueError: If the DataFrame is empty.
    """
    output_path = Path(output_path)

    if df.empty:
        raise ValueError("Cannot save an empty DataFrame.")

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} rows to: {output_path}")



     
     

          


