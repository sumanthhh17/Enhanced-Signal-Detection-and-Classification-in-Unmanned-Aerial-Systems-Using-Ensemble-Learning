import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data (FIXED PATH)
df = pd.read_csv(r"C:\Users\Sumanth\OneDrive\Desktop\mini project\logged_data.csv")

print("="*60)
print("EXPLORATORY DATA ANALYSIS - logged_data.csv")
print("="*60)
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\n" + "="*60)
print("DATA INFO")
print("="*60)
df.info()

print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n" + "="*60)
print("DATA TYPES")
print("="*60)
print(df.dtypes)

# Numeric columns correlation (matplotlib only)
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    print("\n" + "="*60)
    print("CORRELATION MATRIX")
    print("="*60)
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)
    
    # Simple correlation plot
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("EDA COMPLETE!")
print("="*60)
