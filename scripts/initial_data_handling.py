"""
This script performs the following steps to simulate arriving data:
1) takes the full raw data and sets aside 150,000 as final test set
2) From the remaining data, it leaves one class label out ('DAP') 
3) Randomly picks 150,000 samples from the remaining samples for intial work.
4) Returns the left out label, and generates additional splits with same size, 5-10% of data will overlap with our existing data, until all data is included. 
"""

import pandas as pd
from pathlib import Path
import argparse
from typing import Optional
import random

# --- Configuration Constants ---
RAW_DATA_PATH = Path("data/train.csv")
TEST_SET_OUTPUT_PATH = Path("data/test_set.csv")
INITIAL_DATASET_OUTPUT_PATH = Path("data/dataset_initial.csv")
TARGET_COLUMN = 'fertilizer_name'
TEST_SAMPLE_SIZE = 150_000
INITIAL_SAMPLE_SIZE = 150_000
EXCLUDE_LABEL = 'DAP'
BATCH_SIZE = 150_000
MIN_OVERLAP_PERCENTAGE = 0.05 
MAX_OVERLAP_PERCENTAGE = 0.10
BATCHES_OUTPUT_DIR = Path("data/simulated_batches")
LEFT_OUT_DATA_PATH = Path("data/dataset_left_out.csv")


def prepare_initial_datasets(
    raw_path: Path, 
    test_output: Path, 
    initial_output: Path, 
    target_col: str, 
    test_size: int, 
    initial_size: int, 
    exclude_label: str, 
    left_out_output: Path, 
    random_seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Loads the raw CSV data, performs sampling for the permanent test set, 
    generates a filtered development dataset, and returns the dataframe of
    all remaining samples for subsequent batch creation.
    """
    
    # Ensure the output directory exists
    test_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load the full dataset
        print(f"Loading raw data from: {raw_path}")
        df_full = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {raw_path}. Please check the path.")
        return pd.DataFrame()

    # Change the column names to remove typos and spaces
    df_full.columns = [col.strip().lower().replace(" ", "_").replace("temparature", "temperature") for col in df_full.columns]
    df_full.to_csv(raw_path, index=False)  # Save corrected column names back to CSV
    print(f"Raw data loaded successfully with {len(df_full)} rows and columns: {df_full.columns.tolist()}")

    # --- A. Create Permanent Test Set (Step 1) ---
    
    # Randomly sample the test set (this data is now reserved for final evaluation)
    df_test_set = df_full.sample(n=test_size, random_state=random_seed)
    
    # Save the test set
    df_test_set.to_csv(test_output, index=False)
    print(f"SUCCESS: Permanent test set ({len(df_test_set)} rows) saved to {test_output}")

    # --- B. Get Remaining Data ---

    # Get the remaining data (excluding the test set to maintain independence)
    test_indices = df_test_set.index
    df_remaining_for_dev = df_full.drop(test_indices)

    # --- C. Create Initial Development Dataset (Steps 2 & 3) ---

    # Split data into excluded label and non-excluded data
    df_excluded_label = df_remaining_for_dev[df_remaining_for_dev[target_col] == exclude_label]
    df_non_excluded = df_remaining_for_dev[df_remaining_for_dev[target_col] != exclude_label]

    # Sample the initial development dataset from the non-excluded data
    if len(df_non_excluded) < initial_size:
        print(f"WARNING: Only {len(df_non_excluded)} rows available after filtering (excluding {exclude_label}). Using all available rows for initial set.")
        df_initial = df_non_excluded
        # The remaining non_excluded data is now empty
        df_remaining_non_excluded = pd.DataFrame() 
    else:
        df_initial = df_non_excluded.sample(n=initial_size, random_state=random_seed)
        # Data not selected for the initial set
        initial_indices = df_initial.index
        df_remaining_non_excluded = df_non_excluded.drop(initial_indices)


    # Save the initial development set
    df_initial.to_csv(initial_output, index=False)
    print(f"SUCCESS: Initial development dataset ({len(df_initial)} rows, excluding '{exclude_label}') saved to {initial_output}")
    
    # Combine the excluded label data and the remaining non-excluded data
    df_left_out = pd.concat([df_excluded_label, df_remaining_non_excluded])
    
    # Save the full left-out data for step 4
    df_left_out.to_csv(left_out_output, index=False)
    print(f"INFO: All remaining samples ({len(df_left_out)} rows) saved to {left_out_output} for batch generation.")

    return df_left_out


def generate_simulated_batches(
    df_left_out: pd.DataFrame, 
    initial_dataset_path: Path, 
    batch_size: int, 
    min_overlap_percentage: float, 
    max_overlap_percentage: float, 
    output_dir: Path, 
    random_seed: Optional[int] = 42
) -> None:
    """
    Generates subsequent data batches from the left-out dataset, 
    incorporating the left-out label and ensuring a reproducible random overlap
    (between min_overlap_percentage and max_overlap_percentage) with existing data.
    """
    
    if df_left_out.empty:
        print("INFO: No samples left to generate subsequent batches.")
        return

    # Ensure output directory exists and is empty for new batches
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in output_dir.glob('batch_*.csv'):
        file.unlink() # Clear previous batches
    
    # Load the initial dataset to use for overlap
    try:
        df_initial = pd.read_csv(initial_dataset_path)
    except FileNotFoundError:
        print(f"ERROR: Initial dataset not found at {initial_dataset_path}. Cannot create overlapping batches.")
        return
        
    print(f"\n--- Generating Simulated Arriving Data Batches ---")
    print(f"Batch size: {batch_size}. Overlap will be random between {min_overlap_percentage*100:.1f}% and {max_overlap_percentage*100:.1f}% per batch.")

    # The pool for overlap sampling
    df_overlap_pool = df_initial.copy()
    
    # The pool for new samples
    df_new_samples_pool = df_left_out.copy()
    
    batch_num = 1
    while not df_new_samples_pool.empty:
        
        # --- Random Overlap Calculation (The NEW part) ---
        
        # Seed the random number generator uniquely for this batch
        random.seed(random_seed + batch_num) 
        
        # Determine a random overlap percentage for this batch
        overlap_percentage = random.uniform(min_overlap_percentage, max_overlap_percentage)
        overlap_count = int(batch_size * overlap_percentage)
        
        # Ensure we don't try to overlap with more samples than are available in the pool
        overlap_count = min(overlap_count, len(df_overlap_pool))
        
        # 1. Sample the overlap portion from the current overlap pool
        # This simulates data that is 'seen again'
        df_overlap_sample = df_overlap_pool.sample(
            n=overlap_count, 
            replace=False, 
            random_state=random_seed + batch_num # Use the same seed logic for sampling reproducibility
        )
        
        # Determine how many NEW samples are needed for this batch
        needed_new_samples = batch_size - len(df_overlap_sample)
        
        # 2. Sample the new portion from the remaining new samples pool
        df_new_sample = df_new_samples_pool.sample(
            n=min(needed_new_samples, len(df_new_samples_pool)), 
            replace=False, 
            random_state=random_seed + batch_num
        )
        
        # --- Create and Save the Batch ---
        df_batch = pd.concat([df_overlap_sample, df_new_sample]).reset_index(drop=True)
        batch_path = output_dir / f"batch_{batch_num:02d}.csv"
        df_batch.to_csv(batch_path, index=False)
        
        print(f"SUCCESS: Generated batch {batch_num:02d} ({len(df_batch)} rows). Overlap: {len(df_overlap_sample)} rows ({overlap_percentage*100:.2f}%)")
        
        # --- Update Pools for Next Batch ---
        
        # Remove the new samples used in this batch from the new samples pool
        used_indices = df_new_sample.index
        df_new_samples_pool.drop(used_indices, inplace=True)

        # The overlap pool for the next batch grows by the new samples added in this batch.
        df_overlap_pool = pd.concat([df_overlap_pool, df_new_sample]).reset_index(drop=True)
        
        # If we didn't fill the full batch size, we're done.
        if len(df_batch) < batch_size:
            print(f"INFO: All remaining {len(df_new_sample)} samples have been used.")
            break
            
        batch_num += 1


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Prepare initial test and development datasets from raw CSV.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    df_left_out_data = prepare_initial_datasets(
        raw_path=RAW_DATA_PATH,
        test_output=TEST_SET_OUTPUT_PATH,
        initial_output=INITIAL_DATASET_OUTPUT_PATH,
        target_col=TARGET_COLUMN,
        test_size=TEST_SAMPLE_SIZE,
        initial_size=INITIAL_SAMPLE_SIZE,
        exclude_label=EXCLUDE_LABEL,
        left_out_output=LEFT_OUT_DATA_PATH,
        random_seed=args.seed
    )

    generate_simulated_batches(
        df_left_out=df_left_out_data,
        initial_dataset_path=INITIAL_DATASET_OUTPUT_PATH,
        batch_size=BATCH_SIZE,
        min_overlap_percentage=MIN_OVERLAP_PERCENTAGE, # Pass the min/max range
        max_overlap_percentage=MAX_OVERLAP_PERCENTAGE,
        output_dir=BATCHES_OUTPUT_DIR,
        random_seed=args.seed
    )