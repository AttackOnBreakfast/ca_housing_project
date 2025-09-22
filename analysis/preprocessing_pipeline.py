import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# Load the raw training data
raw_train_path = os.path.join("data", "train", "housing_train.csv")
df = pd.read_csv(raw_train_path)

# Separate features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Separate numerical and categorical columns
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Define transformers
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Full preprocessing pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Fit and transform
X_processed = full_pipeline.fit_transform(X)

# Reconstruct DataFrame
X_processed_df = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
)

# Add back the target column
processed_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

# Save processed dataset
output_path = os.path.join("data", "train", "housing_train_processed.csv")
processed_df.to_csv(output_path, index=False)

print(f"✅ Processed dataset saved to {output_path}")