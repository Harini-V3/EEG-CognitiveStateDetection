import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import joblib

df = pd.read_csv("Condes_eeg_feature_extracted.csv")
X = df.drop(columns=["label"])
y = df["label"]

print("Concentration Class Distribution:", Counter(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=20, max_depth=3,
    min_samples_leaf=4, max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nConcentration Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Test Accuracy: {accuracy * 100:.2f}%")

cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")

joblib.dump(model, "model_concentration.pkl")
print("Model saved as 'model_concentration.pkl'")