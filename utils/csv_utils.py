import pandas as pd


def load_csv_safe(file_path):
    """Safely load CSV"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def check_required_columns(df, required_columns):
    """Ensure required columns exist"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def drop_missing_rows(df, required_columns):
    """Drop rows with missing important values"""
    return df.dropna(subset=required_columns)
