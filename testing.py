from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    print(len(df))
    folder_path = Path('./data/train_data') # Use '.' for the current directory

    # Count only files (non-recursive)
    file_count = len([p for p in folder_path.iterdir() if p.is_file()])
    print(f"Number of files in '{folder_path}': {file_count}")