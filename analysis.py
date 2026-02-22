import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
data = pd.read_csv(r"C:\Users\Sumanth\OneDrive\Desktop\mini project\logged_data.csv")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING – Battery Level
# ─────────────────────────────────────────────
data['Battery Level'] = (
    100
    - (data['CPU Usage'] * 0.3)
    - (data['WiFi Strength'].abs() * 0.2)
    - (data['System Load'] * 5)
)
data['Battery Level'] = data['Battery Level'].clip(10, 100).astype(int)

# ─────────────────────────────────────────────
# 3. CLEANING
# ─────────────────────────────────────────────
data.drop(['Air Pressure', 'I/Q Data'], axis=1, inplace=True)
data['Interference Type'].fillna('Unknown', inplace=True)
data.drop(['Latitude', 'Longitude', 'Altitude(m)'], axis=1, inplace=True)

print("After Cleaning – Null Counts:")
print(data.isnull().sum())

# ─────────────────────────────────────────────
# 4. ENCODING
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 5. EDA
# ─────────────────────────────────────────────
print("\n----- EDA SECTION -----")
print("Dataset Shape:", data.shape)
print("Columns:", data.columns.tolist())
print("\nData Types:\n", data.dtypes)
print("\nStatistical Summary:\n", data.describe())

plt.figure(figsize=(6, 4))
sns.countplot(x=data['Device Status'])
plt.title("Device Status Distribution")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 4))
plt.hist(data['Battery Level'], bins=20)
plt.title("Battery Level Distribution")
plt.xlabel("Battery Level"); plt.ylabel("Count")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 4))
plt.hist(data['Signal Strength'], bins=30)
plt.title("Signal Strength Distribution")
plt.xlabel("Signal Strength"); plt.ylabel("Count")
plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(numeric_only=True), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 6. CLASSIFICATION – Device Status
# ─────────────────────────────────────────────
print("\n----- Device Status Classification -----")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier           # ← NEW

X = data.drop(['Device Status', 'Battery Level'], axis=1)
y = data['Device Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Training size:", X_train.shape)
print("Testing size :", X_test.shape)

# Model 1 – Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_accuracy = rf.score(X_test, y_test)
print("Random Forest Accuracy       :", rf_accuracy)

# Model 2 – Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
gb_accuracy = gb.score(X_test, y_test)
print("Gradient Boosting Accuracy   :", gb_accuracy)

# Model 3 – XGBoost  ← NEW
xgb_clf = XGBClassifier(
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
xgb_clf.fit(X_train, y_train)
xgb_accuracy = xgb_clf.score(X_test, y_test)
print("XGBoost Accuracy             :", xgb_accuracy)

# Ensemble – Voting (all 3 models)
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('xgb', xgb_clf)],
    voting='hard'
)
ensemble.fit(X_train, y_train)
ens_accuracy = ensemble.score(X_test, y_test)
print("Ensemble (RF+GB+XGB) Accuracy:", ens_accuracy)

# Confusion Matrix
y_pred = ensemble.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – Ensemble (Classification)")
plt.tight_layout(); plt.show()

# Model Accuracy Comparison
models_clf = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Ensemble']
accuracies  = [rf_accuracy, gb_accuracy, xgb_accuracy, ens_accuracy]

plt.figure(figsize=(9, 5))
bars = plt.bar(models_clf, accuracies, color=['steelblue', 'darkorange', 'green', 'crimson'])
plt.bar_label(bars, fmt='%.4f', padding=3)
plt.title('Model Accuracy Comparison – Device Status')
plt.xlabel('Models'); plt.ylabel('Accuracy')
plt.ylim(0, 1.15)
plt.tight_layout(); plt.show()

# Feature Importance (Random Forest)
importances = rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 7. REGRESSION – Battery Level
# ─────────────────────────────────────────────
print("\n----- Battery Level Prediction -----")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor           # ← NEW

X_battery = data.drop(['Device Status', 'Battery Level'], axis=1)
y_battery  = data['Battery Level']

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_battery, y_battery, test_size=0.3, random_state=42
)

# Model 1 – Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(Xb_train, yb_train)
rf_r2 = rf_reg.score(Xb_test, yb_test)
print("Random Forest Battery R²     :", rf_r2)

# Model 2 – Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(random_state=42)
gb_reg.fit(Xb_train, yb_train)
gb_r2 = gb_reg.score(Xb_test, yb_test)
print("Gradient Boosting Battery R² :", gb_r2)

# Model 3 – XGBoost Regressor  ← NEW
xgb_reg = XGBRegressor(n_estimators=100, random_state=42)
xgb_reg.fit(Xb_train, yb_train)
xgb_r2 = xgb_reg.score(Xb_test, yb_test)
print("XGBoost Battery R²           :", xgb_r2)

# Ensemble via averaging predictions  ← NEW STRATEGY
yb_pred_rf  = rf_reg.predict(Xb_test)
yb_pred_gb  = gb_reg.predict(Xb_test)
yb_pred_xgb = xgb_reg.predict(Xb_test)

yb_pred_ensemble = (yb_pred_rf + yb_pred_gb + yb_pred_xgb) / 3

from sklearn.metrics import r2_score
ens_r2  = r2_score(yb_test, yb_pred_ensemble)
ens_mae = mean_absolute_error(yb_test, yb_pred_ensemble)
ens_mse = mean_squared_error(yb_test, yb_pred_ensemble)
print(f"Ensemble Battery R²          : {ens_r2:.4f}")
print(f"Ensemble Battery MAE         : {ens_mae:.4f}")
print(f"Ensemble Battery MSE         : {ens_mse:.4f}")

# R² Comparison Chart
models_reg = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Ensemble\n(Avg)']
r2_scores  = [rf_r2, gb_r2, xgb_r2, ens_r2]

plt.figure(figsize=(9, 5))
bars = plt.bar(models_reg, r2_scores, color=['steelblue', 'darkorange', 'green', 'crimson'])
plt.bar_label(bars, fmt='%.4f', padding=3)
plt.title('Model R² Comparison – Battery Level')
plt.xlabel('Models'); plt.ylabel('R² Score')
plt.ylim(0, 1.15)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 8. SAVE MODELS
# ─────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))

joblib.dump(ensemble,  os.path.join(base_dir, "device_status_model.pkl"))
joblib.dump(rf_reg,    os.path.join(base_dir, "battery_model.pkl"))
joblib.dump(xgb_clf,   os.path.join(base_dir, "xgb_classifier.pkl"))  # ← NEW
joblib.dump(xgb_reg,   os.path.join(base_dir, "xgb_regressor.pkl"))   # ← NEW

print("\nAll models saved successfully.")
