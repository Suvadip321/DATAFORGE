import pandas as pd
import pathlib
from typing import Tuple, Optional

def load_csv(file_path: pathlib.Path, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Loads csv data from a file path and separates features(X) from the target(y) if the target column is provided as argument.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file_path}. Check the path or name.")
    except Exception as e:
        raise Exception(f"An error occurred during file loading: {e}")
    
    if target_column is None:
        return df, None
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

if __name__ == '__main__':
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent
    TEST_DATA_PATH = PROJECT_ROOT / "sample_data.csv"

    try:
        # test 1
        df_full, _ = load_csv(TEST_DATA_PATH)
        print(f"\n[Test 1: Success] load_csv full return (EDA Mode).")
        print(f"   -> Full DataFrame shape: {df_full.shape}")

        # test 2
        X_, y_ = load_csv(TEST_DATA_PATH, target_column='Target') 
        print(f"\n[Test 2: Success] load_csv split (ML Mode).")
        print(f"   -> Features (X) columns: {X_.columns.tolist()}")
        
        # test 3
        print("\n[Test 3: Checking Validation Error...]")
        try:
            load_csv(TEST_DATA_PATH, target_column='NON_EXISTENT_COLUMN')
        except ValueError as ve:
            print(f"   -> Validation Passed: Caught expected error: {ve}")
            
    except Exception as e:
        print(f"\n--- CRITICAL TEST FAILURE! --- \nDetails: {e}")