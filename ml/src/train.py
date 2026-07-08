"""Train the 'worth reading' classifier on the labeled sentence dataset."""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from features import extract_features

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "labeled_sentences.csv"
MODEL_DIR = ROOT / "model"

# Load labeled data.
df = pd.read_csv(DATA_CSV)

# Extract features for every sentence.
rows = []
for _, row in df.iterrows():
    feats = extract_features(row["sentence"])
    if feats:
        feats["label"] = row["label"]
        rows.append(feats)

data = pd.DataFrame(rows)
X = data.drop("label", axis=1)
y = data["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train — start conservative. Raise max_depth to watch overfitting appear.
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)

print(f"Train accuracy: {train_acc:.3f}")
print(f"Val accuracy:   {val_acc:.3f}")
print(f"Gap:            {train_acc - val_acc:.3f}")
# If gap > 0.15, you're overfitting. Reduce max_depth.

print("\n", classification_report(y_val, model.predict(X_val)))

# See which features mattered most.
feature_importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\nFeature importance:\n", feature_importance)

# Persist the feature column order (the extension must feed features in this
# exact order) and the trained model (for the ONNX export step).
MODEL_DIR.mkdir(parents=True, exist_ok=True)
with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(list(X.columns), f)
with open(MODEL_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\nSaved feature_names.json and model.pkl to {MODEL_DIR}")
