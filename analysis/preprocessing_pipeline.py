import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Get the path of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

raw_train_path = os.path.join(DATA_DIR, "train", "housing_train.csv")
output_path = os.path.join(DATA_DIR, "train", "housing_train_processed.csv")

# Load data
df = pd.read_csv(raw_train_path)

# Separate features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Identify numeric and categorical features
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

X_processed = full_pipeline.fit_transform(X)

# Convert to DataFrame
X_processed_df = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
)

# Add target back
processed_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

# Save
processed_df.to_csv(output_path, index=False)
print(f"✅ Processed dataset saved to {output_path}")