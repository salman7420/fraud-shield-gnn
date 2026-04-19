from src.utils.data_configs import  BASE_TRAIN, BASE_TEST,BASE_VAL ,V1_TEST, V1_TRAIN, V1_VAL
from src.utils.logger import get_logger
from src.data_ingestion.load_data import load_data, save_data

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = get_logger(__name__) 


# ── Load base data  ───────────────────────────────────────────

v1_train = load_data(BASE_TRAIN)
v1_test = load_data(BASE_TEST)
v1_val = load_data(BASE_VAL)

# ── Step 2: Drop TransactionID ────────────────────────────────

def drop_id_column(df: pd.DataFrame, col: str = "TransactionID") -> pd.DataFrame:
    if col in df.columns:
        df = df.drop(columns=[col])
        logger.info(f"Dropped column: '{col}'")
    else:
        logger.warning(f"Column '{col}' not found — skipping drop")
    return df

v1_train = drop_id_column(v1_train)
v1_val   = drop_id_column(v1_val)
v1_test  = drop_id_column(v1_test)

# ── Step 3: Separate features and target ─────────────────────

def separate_X_y(df: pd.DataFrame, target: str = "isFraud"):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    logger.info(f"Features shape: {X.shape} | Target shape: {y.shape}")
    logger.info(f"Fraud rate: {y.mean():.4f} ({y.mean() * 100:.2f}%)")
    
    return X, y


X_train, y_train = separate_X_y(v1_train)
X_val,   y_val   = separate_X_y(v1_val)
X_test,  y_test  = separate_X_y(v1_test)

# ── Step 4: Identify column types ─────────────────────────────
def identify_column_types(X: pd.DataFrame):
    """
    Fit on X_train only.
    Returns two lists: numeric columns and categorical columns.
    """
    numeric_cols     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    logger.info(f"Numeric columns     : {len(numeric_cols)}")
    logger.info(f"Categorical columns : {len(categorical_cols)}")
    logger.info(f"Categorical cols    : {categorical_cols}")

    return numeric_cols, categorical_cols


# Fit on train only — val and test will use these same lists
numeric_cols, categorical_cols = identify_column_types(X_train)

# ── Step 5: Fill numeric nulls with -999 ──────────────────────
NULL_NUMERIC = -999

def fill_numeric_nulls(X: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    before = X[numeric_cols].isnull().sum().sum()
    X[numeric_cols] = X[numeric_cols].fillna(NULL_NUMERIC)
    after  = X[numeric_cols].isnull().sum().sum()

    logger.info(f"Numeric nulls before: {before:,} | after: {after:,}")
    return X


X_train = fill_numeric_nulls(X_train, numeric_cols)
X_val   = fill_numeric_nulls(X_val,   numeric_cols)
X_test  = fill_numeric_nulls(X_test,  numeric_cols)

# ── Step 6: Fill categorical nulls with "Unknown" ─────────────
NULL_CATEGORICAL = "Unknown"

def fill_categorical_nulls(X: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    before = X[categorical_cols].isnull().sum().sum()
    X[categorical_cols] = X[categorical_cols].fillna(NULL_CATEGORICAL)
    after  = X[categorical_cols].isnull().sum().sum()

    logger.info(f"Categorical nulls before: {before:,} | after: {after:,}")
    return X


X_train = fill_categorical_nulls(X_train, categorical_cols)
X_val   = fill_categorical_nulls(X_val,   categorical_cols)
X_test  = fill_categorical_nulls(X_test,  categorical_cols)


# ── Step 7: Label encode categorical columns ───────────────────


def fit_label_encoders(X_train: pd.DataFrame, categorical_cols: list) -> dict:
    """
    Fit one LabelEncoder per categorical column on TRAIN only.
    Returns a dict of {col_name: fitted_encoder}.
    """
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_train[col])
        encoders[col] = le
        logger.info(f"Fitted encoder for '{col}' | {len(le.classes_)} unique classes")

    logger.info(f"Total encoders fitted: {len(encoders)}")
    return encoders


def apply_label_encoders(X: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Transform categorical columns using already-fitted encoders.
    Handles unseen categories by mapping them to 'Unknown'.
    """
    for col, le in encoders.items():
        known_classes = set(le.classes_)

        # Replace any unseen category with "Unknown" before transforming
        unseen = ~X[col].isin(known_classes)
        if unseen.sum() > 0:
            logger.warning(
                f"'{col}': {unseen.sum():,} unseen categories → replaced with 'Unknown'"
            )
            X[col] = X[col].where(X[col].isin(known_classes), other="Unknown")

        X[col] = le.transform(X[col])

    return X


# ── Fit on train, transform all three ─────────────────────────
encoders = fit_label_encoders(X_train, categorical_cols)

X_train = apply_label_encoders(X_train, encoders)
X_val   = apply_label_encoders(X_val,   encoders)
X_test  = apply_label_encoders(X_test,  encoders)

# ── Step 8: Save v1 processed data ────────────────────────────

# Recombine X and y before saving
train_out = pd.concat([X_train, y_train], axis=1)
val_out   = pd.concat([X_val,   y_val],   axis=1)
test_out  = pd.concat([X_test,  y_test],  axis=1)

save_data(train_out, V1_TRAIN)
save_data(val_out,   V1_VAL)
save_data(test_out,  V1_TEST)
