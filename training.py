import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os

# === Path to your CSV file ===
DATA_PATH = "diabetes.csv"  # Change to full path if not in same folder

# === Detect if CSV has headers ===
with open(DATA_PATH, 'r') as f:
    first_line = f.readline().strip().split(',')
    expected_headers = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    has_header = all(col in expected_headers for col in first_line)

# === Load dataset ===
if has_header:
    data = pd.read_csv(DATA_PATH)
else:
    data = pd.read_csv(DATA_PATH, names=expected_headers)

# === Split features and target ===
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Models to try ===
models = {
    'LogisticRegression': LogisticRegression(max_iter=200),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'DecisionTree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier()
}

best_model = None
best_f1 = 0
best_name = None
metrics = {}

# === Train and evaluate each model ===
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics[name] = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4)
    }

    print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} with F1 Score: {best_f1:.4f}")

# === Save best model ===
joblib.dump(best_model, 'diabetes_model.pkl')

# === Save metrics to JSON ===
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("\nBest model and metrics have been saved successfully!")
