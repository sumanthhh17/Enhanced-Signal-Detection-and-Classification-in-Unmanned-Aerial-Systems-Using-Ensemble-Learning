import pandas as pd
pip install seaborn
from pathlib import Path
data = pd.read_csv(Path("C:/Users/Sumanth/OneDrive/Desktop/mini project/logged_data.csv"))

# Drop useless columns
data.drop(['Air Pressure', 'I/Q Data'], axis=1, inplace=True)

# Fill missing categorical values
data['Interference Type'].fillna('Unknown', inplace=True)

# Optional – drop constant location columns
data.drop(['Latitude', 'Longitude', 'Altitude(m)'], axis=1, inplace=True)

print("After Cleaning:")
print(data.isnull().sum())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_cols = [
    'Modulation',
    'Location',
    'Device Type',
    'Antenna Type',
    'Weather Condition',
    'Interference Type',
    'Device Status'
]

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

data.drop(['Timestamp'], axis=1, inplace=True)
# Separate input (X) and output (y)
X = data.drop('Device Status', axis=1)
y = data['Device Status']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

rf_accuracy = rf.score(X_test, y_test)
print("Random Forest Accuracy:", rf_accuracy)
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

gb_accuracy = gb.score(X_test, y_test)
print("Gradient Boosting Accuracy:", gb_accuracy)
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='hard'
)

ensemble.fit(X_train, y_train)
ens_accuracy = ensemble.score(X_test, y_test)

print("Ensemble Accuracy:", ens_accuracy)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = ensemble.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()
import matplotlib.pyplot as plt

# Accuracy values
models = ['Random Forest', 'Gradient Boosting', 'Ensemble']
accuracies = [rf_accuracy, gb_accuracy, ens_accuracy]

plt.figure(figsize=(8,5))
plt.bar(models, accuracies)
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)   # accuracy range
plt.show()
import matplotlib.pyplot as plt

importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()
print("\n----- Battery Level Prediction -----")

# Target = Battery Level
X_battery = data.drop('Battery Level', axis=1)
y_battery = data['Battery Level']

from sklearn.model_selection import train_test_split

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_battery, y_battery, test_size=0.3, random_state=42
)

# Model 1 - Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(Xb_train, yb_train)

rf_score = rf_reg.score(Xb_test, yb_test)
print("Random Forest Battery R2:", rf_score)

# Model 2 - Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

gb_reg = GradientBoostingRegressor()
gb_reg.fit(Xb_train, yb_train)

gb_score = gb_reg.score(Xb_test, yb_test)
print("Gradient Boosting Battery R2:", gb_score)
import joblib

joblib.dump(ensemble, "device_status_model.pkl")
joblib.dump(rf_reg, "battery_model.pkl")

print("Models saved successfully")