
from sklearn.datasets import load_breast_cancer
import pandas as pd
from pathlib import Path

# Load dataset
data = load_breast_cancer()

# Create DataFrame with 30 feature columns
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["target"] = data.target

# Check feature count
feature_count = len(data.feature_names)

if feature_count != 30:
    raise ValueError(f"Expected 30 features, found {feature_count}")

# Check for missing values
if df.isnull().values.any():
    raise ValueError("Dataset contains null values!")

# Create output directory
output_path = Path("data/raw")
output_path.mkdir(parents=True, exist_ok=True)

# Export CSV
csv_path = output_path / "breast_cancer.csv"
df.to_csv(csv_path, index=False)

print("Dataset exported successfully!")
print("Dataset shape:", df.shape)
print("Feature count:", feature_count)
print("Null values:", df.isnull().sum().sum())
print("Target distribution:")
print(df["target"].value_counts())