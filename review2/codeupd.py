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

# Artificial Battery Level generation
data['Battery Level'] = (
    100
    - (data['CPU Usage'] * 0.3)
    - (data['WiFi Strength'].abs() * 0.2)
    - (data['System Load'] * 5)
)

data['Battery Level'] = data['Battery Level'].clip(10, 100).astype(int)

# -------------------------------
# 4. CLEANING
# -------------------------------
print("\n===== DATA CLEANING =====")

# Drop irrelevant / heavy columns
drop_cols = ['Air Pressure', 'I/Q Data', 'Latitude', 'Longitude', 'Altitude(m)']
for col in drop_cols:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

# Fill missing categorical values
if 'Interference Type' in data.columns:
    data['Interference Type'] = data['Interference Type'].fillna('Unknown')

# -------------------------------
# 5. ENCODING
# -------------------------------
print("\n===== ENCODING =====")

from sklearn.preprocessing import LabelEncoder

encoders = {}

categorical_cols = [
    'Modulation', 'Location', 'Device Type',
    'Antenna Type', 'Weather Condition',
    'Interference Type', 'Device Status'
]

for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

# Drop timestamp if present
if 'Timestamp' in data.columns:
    data.drop(['Timestamp'], axis=1, inplace=True)

# -------------------------------
# 6. VISUALIZATION (EDA)
# -------------------------------
print("\n===== DATA DISTRIBUTION =====")

data.hist(figsize=(14,10))
plt.tight_layout()
plt.show()

print("\n===== CORRELATION HEATMAP =====")

plt.figure(figsize=(14,8))
sns.heatmap(data.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------------
# 7. FEATURES & TARGETS
# -------------------------------
print("\n===== FEATURES & TARGETS =====")

X = data.drop(['Device Status', 'Interference Type', 'Battery Level', 'WiFi Strength'], axis=1)

y_class = data[['Device Status', 'Interference Type']]
y_reg   = data[['Battery Level', 'WiFi Strength']]

print("Feature Columns:", list(X.columns))

# -------------------------------
# 8. SPLIT
# -------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)

# -------------------------------
# 9. ENSEMBLE MODELS
# -------------------------------
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# Classification
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)

base_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('gb', gb_clf)],
    voting='hard'
)

multi_clf = MultiOutputClassifier(base_clf)

# Regression
rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
gb_reg = GradientBoostingRegressor(random_state=42)

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

# -------------------------------
# 12. PREDICT
# -------------------------------
print("\n===== PREDICTIONS =====")

predictions = hybrid_model.predict(X_test)

class_pred = predictions[:, :2]
reg_pred   = predictions[:, 2:]

# Convert classification predictions to int
class_pred = class_pred.astype(int)

# -------------------------------
# 13. EVALUATION
# -------------------------------
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

device_acc = accuracy_score(y_class_test.iloc[:, 0], class_pred[:, 0])
inter_acc  = accuracy_score(y_class_test.iloc[:, 1], class_pred[:, 1])

battery_r2 = r2_score(y_reg_test.iloc[:, 0], reg_pred[:, 0])
wifi_r2    = r2_score(y_reg_test.iloc[:, 1], reg_pred[:, 1])

battery_mae = mean_absolute_error(y_reg_test.iloc[:, 0], reg_pred[:, 0])
wifi_mae    = mean_absolute_error(y_reg_test.iloc[:, 1], reg_pred[:, 1])

print("\n===== FINAL RESULTS =====")
print(f"Device Accuracy        : {device_acc:.4f}")
print(f"Interference Accuracy  : {inter_acc:.4f}")
print(f"Battery R2 Score       : {battery_r2:.4f}")
print(f"WiFi R2 Score          : {wifi_r2:.4f}")
print(f"Battery MAE            : {battery_mae:.4f}")
print(f"WiFi MAE               : {wifi_mae:.4f}")

# -------------------------------
# 14. SAMPLE DECODED PREDICTIONS
# -------------------------------
print("\n===== SAMPLE DECODED PREDICTIONS =====")

device_labels = encoders['Device Status'].inverse_transform(class_pred[:, 0])
interference_labels = encoders['Interference Type'].inverse_transform(class_pred[:, 1])

sample_results = pd.DataFrame({
    "Predicted Device Status": device_labels[:10],
    "Predicted Interference Type": interference_labels[:10],
    "Predicted Battery Level": reg_pred[:10, 0].round(2),
    "Predicted WiFi Strength": reg_pred[:10, 1].round(2)
})

print(sample_results)

# -------------------------------
# 15. FEATURE IMPORTANCE
# -------------------------------
print("\n===== FEATURE IMPORTANCE =====")

target_names = ['Battery Level', 'WiFi Strength']

for i, target in enumerate(target_names):
    trained_voting_reg = hybrid_model.reg_model.estimators_[i]
    trained_rf_reg = trained_voting_reg.named_estimators_['rf']

    importances = trained_rf_reg.feature_importances_
    features = X.columns

    plt.figure(figsize=(8,5))
    plt.barh(features, importances)
    plt.title(f"Feature Importance for {target}")
    plt.xlabel("Importance Score")
    plt.show()

# -------------------------------
# 16. MODEL COMPARISON GRAPH
# -------------------------------
print("\n===== MODEL COMPARISON =====")

# Compare for Device Status prediction
rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
gb_temp = GradientBoostingClassifier(random_state=42)
ens_temp = VotingClassifier(
    estimators=[('rf', rf_temp), ('gb', gb_temp)],
    voting='hard'
)

rf_temp.fit(X_train, y_class_train.iloc[:, 0])
gb_temp.fit(X_train, y_class_train.iloc[:, 0])
ens_temp.fit(X_train, y_class_train.iloc[:, 0])

rf_acc = accuracy_score(y_class_test.iloc[:, 0], rf_temp.predict(X_test))
gb_acc = accuracy_score(y_class_test.iloc[:, 0], gb_temp.predict(X_test))
ens_acc = accuracy_score(y_class_test.iloc[:, 0], ens_temp.predict(X_test))

models = ['Random Forest', 'Gradient Boosting', 'Ensemble']
scores = [rf_acc, gb_acc, ens_acc]

plt.figure(figsize=(7,5))
plt.bar(models, scores)
plt.title("Device Status Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# -------------------------------
# 17. SAVE MODEL + ENCODERS
# -------------------------------
print("\n===== SAVING MODEL =====")

base_dir = os.path.dirname(os.path.abspath(__file__))

joblib.dump(hybrid_model, os.path.join(base_dir, "hybrid_model.pkl"))
joblib.dump(encoders, os.path.join(base_dir, "label_encoders.pkl"))

print("Hybrid model saved successfully")
print("Encoders saved successfully")