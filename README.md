# 🏠 California Housing Project

A complete machine learning pipeline for predicting California housing prices using scikit-learn. This project demonstrates best practices in data science workflow, including modular preprocessing, exploratory data analysis (EDA), and model evaluation.

---

## 🧠 Project Overview

This project is structured around the California housing dataset and includes:

- 🧹 **Initial & Exploratory Data Analysis**: Investigating feature distributions, missing values, and geographic trends.
- ⚙️ **Preprocessing Pipeline**: Built with `scikit-learn` using `ColumnTransformer` and `Pipeline` for scalable transformations.
- 🤖 **Model Training**: Implemented and compared several models:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Support Vector Regression (SVR)
- 📈 **Evaluation & Tuning**:
  - Cross-validation performance analysis
  - Hyperparameter tuning using `GridSearchCV`
  - Final model export with `joblib`

---
## 📂 Directory Overview

ca_housing_project/
├── analysis/
│   ├── eda.ipynb
│   ├── ida.ipynb
│   └── preprocessing_pipeline.py
├── data/
│   ├── raw/
│   │   └── housing.csv
│   ├── train/
│   │   ├── housing_train.csv
│   │   └── housing_train_processed.csv
│   └── test/
│       └── housing_test.csv
├── images/
│   └── histograms.png
├── models/
│   ├── LinearRegression.ipynb
│   ├── DecisionTree.ipynb
│   ├── RandomForest.ipynb
│   └── SVR.ipynb
├── .gitignore
├── README.md
└── requirements.txt

---
## 🔁 Workflow Summary

1. **Run `ida.ipynb`**  
   - Loads raw data and analyzes types, distributions, and missing values  
   - Performs stratified train/test split  
   - Saves to `data/train/housing_train.csv` and `data/test/housing_test.csv`

2. **Run `eda.ipynb`**  
   - Visualizes features, correlations, and engineered variables  
   - Outputs `housing_train_processed.csv` with 24 features

3. **Run `preprocessing_pipeline.py`**  
   - Programmatic version of EDA preprocessing  
   - Uses pipelines to clean and transform data  
   - Saves processed dataset

4. **Train models in `/models/*.ipynb`**  
   - Each notebook handles loading, training, CV, tuning, and saving

---

## 📂 Files & Purpose

- `ida.ipynb`: Initial data loading, splitting, and export
- `eda.ipynb`: Exploratory visualizations and feature engineering
- `preprocessing_pipeline.py`: Standalone script for full preprocessing
- `models/*.ipynb`: Separate notebooks for each trained model
- `images/`: All plots used in analysis and EDA
- `data/`: Raw, train, test, and processed CSVs

---

## 💾 Preprocessing (CLI)

To generate the processed training data from the raw training split:

```bash
python analysis/preprocessing_pipeline.py