import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv(r"C:\Users\Sumanth\OneDrive\Desktop\mini project\logged_data.csv")

print("DEEPER EDA ANALYSIS")
print(f"Total rows: {len(df)}, Columns: {df.columns.tolist()}")

# 1. DUPLICATES CHECK
dups = df.duplicated().sum()
print(f"\nDuplicates: {dups}")

# 2. OUTLIERS DETECTION (for numeric columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nNumeric columns: {list(numeric_cols)}")

if len(numeric_cols) > 0:
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        print(f"{col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")

# 3. CATEGORICAL ANALYSIS (if any)
cat_cols = df.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    print(f"\nCategorical columns: {list(cat_cols)}")
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head())

# 4. MOST CORRELATED FEATURES (for modeling)
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(i,j) for i in upper.index for j in upper.columns if upper.loc[i,j] > 0.7]
    print(f"\nHigh correlations (>0.7): {high_corr}")

print("\n" + "="*50)
print("RECOMMENDATIONS:")
print("1. Remove duplicates if any")
print("2. Handle outliers based on domain knowledge")
print("3. Encode categorical variables for modeling")
print("4. Feature engineering (ratios, lags, etc.)")
print("5. Split train/test")
print("="*50)
