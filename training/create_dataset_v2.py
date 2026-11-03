import pandas as pd
from pathlib import Path

# Load Version 1
file_path = Path("data/raw/breast_cancer.csv")
df = pd.read_csv(file_path)

# Check original dataset
if len(df) != 569:
    raise ValueError(f"Expected 569 rows, found {len(df)}")

# Randomly remove 30 rows
removed_rows = df.sample(n=30, random_state=42).index
df_v2 = df.drop(removed_rows).reset_index(drop=True)

# Validate Version 2
if len(df_v2) != 539:
    raise ValueError("Version 2 must contain 539 rows")

if len(df_v2.columns) != 31:
    raise ValueError("Expected 30 features and 1 target column")

if df_v2.isnull().values.any():
    raise ValueError("Dataset contains null values")

# Save Version 2
df_v2.to_csv(file_path, index=False)

print("Version 2 created successfully!")
print("Rows:", len(df_v2))
print("Columns:", len(df_v2.columns))
print("Features:", len(df_v2.columns) - 1)
print("Null values:", df_v2.isnull().sum().sum())
