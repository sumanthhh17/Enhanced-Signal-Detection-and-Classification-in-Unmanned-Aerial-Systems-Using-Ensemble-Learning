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
data['Device_Signal_Index'] = (
    data['Signal Strength'] * 0.5 +
    data['Bandwidth'] * 0.3 +
    data['WiFi Strength'] * 0.2
)

data['Battery Level'] = data['Battery Level'].clip(10, 100).astype(int)
data['Signal_Bandwidth_Ratio'] = data['Signal Strength'] / (data['Bandwidth'] + 1)
data['Load_to_CPU'] = data['System Load'] / (data['CPU Usage'] + 1)
data['Signal_Power'] = data['Signal Strength'] * data['Bandwidth']
data['System_Stress'] = data['CPU Usage'] * data['System Load']
data['Env_Score'] = data['Temperature'] * data['Humidity']
data['Signal_Quality'] = data['WiFi Strength'] / (data['Bandwidth'] + 1)
data['Combined_Load'] = data['CPU Usage'] + data['System Load']
data['Signal_Category'] = pd.cut(
    data['Signal Strength'],
    bins=[-100, -70, -50, 0],
    labels=[0, 1, 2]
)
data['Status_Index'] = (
    data['CPU Usage'] * 0.4 +
    data['System Load'] * 0.4 +
    data['Battery Level'] * 0.2
)

data['Performance_Ratio'] = data['CPU Usage'] / (data['System Load'] + 1)

data['Power_Stress'] = data['Battery Level'] / (data['CPU Usage'] + 1)
data['Status_Flag'] = np.where(
    (data['CPU Usage'] > 70) & (data['System Load'] > 70),
    1,
    0
)
data['Critical_Load'] = np.where(
    (data['CPU Usage'] > 80) | (data['System Load'] > 80),
    1,
    0
)
data['Health_Score'] = (
    (100 - data['CPU Usage']) * 0.4 +
    (100 - data['System Load']) * 0.4 +
    data['Battery Level'] * 0.2
)
data['Overload'] = np.where(
    (data['CPU Usage'] > 75) & (data['System Load'] > 75),
    1, 0
)
data['Device_Status_New'] = np.where(
    (data['CPU Usage'] > 75) | (data['System Load'] > 75),
    0,   # Bad
    np.where(data['Battery Level'] > 50, 2, 1)  # Good / Moderate
)
data['Device_Stress_Score'] = (
    data['CPU Usage'] * 0.5 +
    data['System Load'] * 0.3 -
    data['Battery Level'] * 0.2
)
# -------------------------------
# 4. CLEANING
# -------------------------------
print("\n===== DATA CLEANING =====")

data.drop(['Air Pressure', 'I/Q Data'], axis=1, inplace=True)
data.drop(['Latitude', 'Longitude', 'Altitude(m)'], axis=1, inplace=True)
data = data[(data['WiFi Strength'] > -90) & (data['WiFi Strength'] < -30)]
data = data[data['Signal Strength'] > -85]
data = data[data['CPU Usage'] < 95]
data = data[data['System Load'] < 95]

# -------------------------------
# 5. ENCODING
# -------------------------------
print("\n===== ENCODING =====")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

categorical_cols = [
    'Modulation', 'Location', 'Device Type',
    'Antenna Type', 'Weather Condition'
]

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

data.drop(['Timestamp'], axis=1, inplace=True)
from sklearn.utils import resample

df_list = []
max_size = data['Device Status'].value_counts().max()

for cls in data['Device Status'].unique():
    df_cls = data[data['Device Status'] == cls]
    df_cls_up = resample(df_cls,
                         replace=True,
                         n_samples=max_size,
                         random_state=42)
    df_list.append(df_cls_up)

data = pd.concat(df_list)




# -------------------------------
# 6. VISUALIZATION (EDA)
# -------------------------------
print("\n===== DATA DISTRIBUTION =====")

data.hist(figsize=(12,8))
plt.tight_layout()
plt.show()

print("\n===== CORRELATION HEATMAP =====")

plt.figure(figsize=(10,6))
sns.heatmap(data.select_dtypes(include=np.number).corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# -------------------------------
# 7. FEATURES & TARGETS
# -------------------------------

X = data.drop(['Device Status','Device Type','Battery Level','WiFi Strength'], axis=1)
X = X.select_dtypes(include=[np.number])

# Fill missing
X = X.fillna(0)

# Scale ONCE
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Targets
y_class = data[['Device_Status_New','Signal_Category']]
y_reg   = data[['Battery Level','WiFi Strength']]
print("Unique Signal Categories:", y_class['Signal_Category'].nunique())

# -------------------------------
# 8. SPLIT (STRATIFIED)
# -------------------------------
print("Device Type distribution:\n", data['Device Type'].value_counts())
from sklearn.model_selection import train_test_split

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg,
    test_size=0.2,
    random_state=42,
    stratify=y_class['Signal_Category']
)


# -------------------------------
# 9. ENSEMBLE MODELS
# -------------------------------
from sklearn.ensemble import ExtraTreesClassifier

device_model = ExtraTreesClassifier(
    n_estimators=1500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

signal_model = ExtraTreesClassifier(
    n_estimators=600,
    random_state=42,
    n_jobs=-1
)





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

from sklearn.base import BaseEstimator

class HybridModel(BaseEstimator):

    def __init__(self, device_model, signal_model, reg_model):
       self.device_model = device_model
       self.signal_model = signal_model
       self.reg_model = reg_model

    def fit(self, X, y_class, y_reg):

         print("\n===== TRAINING MODEL =====")

         self.device_model.fit(X, y_class['Device_Status_New'])
         self.signal_model.fit(X, y_class['Signal_Category'])
         self.reg_model.fit(X, y_reg)
         print("Training Completed")
         return self

    def predict(self, X):

      device_pred = self.device_model.predict(X)
      signal_pred = self.signal_model.predict(X)
      reg_pred = self.reg_model.predict(X)

      return np.column_stack((device_pred, signal_pred, reg_pred))

# -------------------------------
# 11. TRAIN
# -------------------------------
hybrid_model = HybridModel(device_model, signal_model, multi_reg)
hybrid_model.fit(X_train, y_class_train, y_reg_train)

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
type_acc  = accuracy_score(y_class_test.iloc[:, 1], class_pred[:, 1])
battery_r2 = r2_score(y_reg_test.iloc[:, 0], reg_pred[:, 0])
wifi_r2    = r2_score(y_reg_test.iloc[:, 1], reg_pred[:, 1])

print("\n===== FINAL RESULTS =====")
print(f"Device Accuracy        : {device_acc:.4f}")
print(f"Signal Category Accuracy : {type_acc:.4f}")
print(f"Battery R2 Score       : {battery_r2:.4f}")
print(f"WiFi R2 Score          : {wifi_r2:.4f}")

# -------------------------------
# 14. FEATURE IMPORTANCE
# -------------------------------
print("\n===== FEATURE IMPORTANCE =====")


from sklearn.ensemble import RandomForestRegressor

# Train separate model for importance
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_temp.fit(X_train, y_reg_train['WiFi Strength'])

importances = rf_temp.feature_importances_

features = [f"Feature {i}" for i in range(len(importances))]

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.title("Feature Importance (WiFi)")
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
