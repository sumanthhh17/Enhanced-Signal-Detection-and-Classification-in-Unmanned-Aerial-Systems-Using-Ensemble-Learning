import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# -------------------------------
# 1. LOAD DATA
# -------------------------------
print("\n===== LOADING DATA =====")
data = pd.read_csv(r"C:\Users\Sumanth\OneDrive\Desktop\minor project\logged_data.csv")

# -------------------------------
# 2. BASIC EDA
# -------------------------------
print("\n===== DATASET INFO =====")
print(data.info())

print("\n===== FIRST 5 ROWS =====")
print(data.head())

print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------
print("\n===== FEATURE ENGINEERING =====")

data['Battery Level'] = (
    100
    - (data['CPU Usage'] * 0.3)
    - (data['WiFi Strength'].abs() * 0.2)
    - (data['System Load'] * 5)
)

data['Battery Level'] = data['Battery Level'].clip(10, 100).astype(int)
data['Signal_Bandwidth_Ratio'] = data['Signal Strength'] / (data['Bandwidth'] + 1)
data['Load_to_CPU'] = data['System Load'] / (data['CPU Usage'] + 1)
data['Signal_to_Noise'] = data['Signal Strength'] / (data['WiFi Strength'].abs() + 1)

# -------------------------------
# 4. CLEANING
# -------------------------------
print("\n===== DATA CLEANING =====")

data.drop(['Air Pressure', 'I/Q Data'], axis=1, inplace=True)
data['Interference Type'] = data['Interference Type'].fillna('Unknown')
data.drop(['Latitude', 'Longitude', 'Altitude(m)'], axis=1, inplace=True)
data = data[(data['WiFi Strength'] > -90) & (data['WiFi Strength'] < -30)]

# -------------------------------
# 5. ENCODING
# -------------------------------
print("\n===== ENCODING =====")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

categorical_cols = [
    'Modulation', 'Location', 'Device Type',
    'Antenna Type', 'Weather Condition',
    'Interference Type', 'Device Status'
]

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

data.drop(['Timestamp'], axis=1, inplace=True)
from sklearn.utils import resample

df_majority = data[data['Interference Type'] == data['Interference Type'].mode()[0]]
df_minority = data[data['Interference Type'] != data['Interference Type'].mode()[0]]

df_minority_upsampled = resample(df_minority,
                                replace=True,
                                n_samples=len(df_majority),
                                random_state=42)

data = pd.concat([df_majority, df_minority_upsampled])

# -------------------------------
# 6. VISUALIZATION (EDA)
# -------------------------------
print("\n===== DATA DISTRIBUTION =====")

data.hist(figsize=(12,8))
plt.tight_layout()
plt.show()

print("\n===== CORRELATION HEATMAP =====")

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------------
# 7. FEATURES & TARGETS
# -------------------------------
# -------------------------------
# FEATURES + CLEANING + SELECTION
# -------------------------------

# FEATURES
X = data.drop(['Device Status','Interference Type','Battery Level','WiFi Strength'], axis=1)

# REMOVE LOW VARIANCE FEATURES
# Remove low variance features (better threshold)
X = X.loc[:, X.var() > 0.01]
X = X.fillna(0)
# TARGETS
y_class = data[['Device Status','Interference Type']]
y_reg   = data[['Battery Level','WiFi Strength']]

# FEATURE SELECTION
from sklearn.feature_selection import SelectKBest, f_regression

k = min(15, X.shape[1])
selector = SelectKBest(score_func=f_regression, k=k)
X = selector.fit_transform(X, y_reg['WiFi Strength'])
# Use regression targets
X = selector.fit_transform(X, y_reg.mean(axis=1))

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
# -------------------------------
# 8. SPLIT
# -------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

_, _, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)


# -------------------------------
# 9. ENSEMBLE MODELS
# -------------------------------
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# Classification

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.multioutput import MultiOutputClassifier

rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)

et_clf = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42
)

base_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('et', et_clf)],
    voting='soft'
)

multi_clf = MultiOutputClassifier(base_clf)

# Regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.multioutput import MultiOutputRegressor

rf_reg = RandomForestRegressor(
    n_estimators=300,   # more trees
    max_depth=12,       # control overfitting
    random_state=42
)

gb_reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

base_reg = VotingRegressor(
    estimators=[('rf', rf_reg), ('gb', gb_reg)]
)

multi_reg = MultiOutputRegressor(base_reg)

# -------------------------------
# 10. HYBRID MODEL
# -------------------------------
from sklearn.base import BaseEstimator

class HybridModel(BaseEstimator):
    
    def __init__(self, clf_model, reg_model):
        self.clf_model = clf_model
        self.reg_model = reg_model

    def fit(self, X, y_class, y_reg):
        print("\n===== TRAINING MODEL =====")
        self.clf_model.fit(X, y_class)
        self.reg_model.fit(X, y_reg)
        print("Training Completed")
        return self

    def predict(self, X):
        class_pred = self.clf_model.predict(X)
        reg_pred   = self.reg_model.predict(X)
        return np.hstack((class_pred, reg_pred))

# -------------------------------
# 11. TRAIN
# -------------------------------
hybrid_model = HybridModel(multi_clf, multi_reg)
hybrid_model.fit(X_train, y_class_train, y_reg_train)
print("\n===== FEATURE IMPORTANCE =====")

# Access regression model inside hybrid
rf_model = hybrid_model.reg_model.estimators_[1].estimators_[0]

importances = rf_model.feature_importances_

print(importances)

# -------------------------------
# 12. PREDICT
# -------------------------------
predictions = hybrid_model.predict(X_test)

class_pred = predictions[:, :2]
reg_pred   = predictions[:, 2:]

# -------------------------------
# 13. EVALUATION
# -------------------------------
from sklearn.metrics import accuracy_score, r2_score

device_acc = accuracy_score(y_class_test.iloc[:, 0], class_pred[:, 0])
inter_acc  = accuracy_score(y_class_test.iloc[:, 1], class_pred[:, 1])
battery_r2 = r2_score(y_reg_test.iloc[:, 0], reg_pred[:, 0])
wifi_r2    = r2_score(y_reg_test.iloc[:, 1], reg_pred[:, 1])

print("\n===== FINAL RESULTS =====")
print(f"Device Accuracy        : {device_acc:.4f}")
print(f"Interference Accuracy  : {inter_acc:.4f}")
print(f"Battery R2 Score       : {battery_r2:.4f}")
print(f"WiFi R2 Score          : {wifi_r2:.4f}")

# -------------------------------
# 14. FEATURE IMPORTANCE
# -------------------------------
print("\n===== FEATURE IMPORTANCE =====")
rf_model = hybrid_model.reg_model.estimators_[1].estimators_[0]
importances = rf_model.feature_importances_

print(importances)

features = selector.get_feature_names_out()

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()

# -------------------------------
# 15. MODEL COMPARISON GRAPH
# -------------------------------
print("\n===== MODEL COMPARISON =====")

models = ['Random Forest', 'Gradient Boosting', 'Ensemble']
scores = [device_acc, device_acc, device_acc]

plt.figure()
plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# -------------------------------
# 16. SAVE MODEL
# -------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(hybrid_model, os.path.join(base_dir, "hybrid_model.pkl"))

print("\nHybrid model saved successfully")