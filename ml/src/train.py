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

# ---------------------------------------------------------------------------
# Learning curve: train vs. validation accuracy as tree depth grows.
# This makes overfitting *visible* — train accuracy climbs toward 1.0 while
# validation accuracy peaks then plateaus/drops. (min_samples_leaf is left
# unset here so deep trees can fully memorize the training set.)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # headless: save to file, don't open a window
import matplotlib.pyplot as plt

depths = [2, 4, 6, 8, 10, 15, 20]
train_scores = []
val_scores = []
for depth in depths:
    m = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    m.fit(X_train, y_train)
    train_scores.append(m.score(X_train, y_train))
    val_scores.append(m.score(X_val, y_val))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, label="Train accuracy", marker="o")
plt.plot(depths, val_scores, label="Validation accuracy", marker="o")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Overfitting curve — train vs. validation accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
curve_path = MODEL_DIR / "overfitting_curve.png"
plt.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.close()

best_i = max(range(len(depths)), key=lambda i: val_scores[i])
print(f"Saved learning curve to {curve_path}")
print(f"  best val accuracy {val_scores[best_i]:.3f} at max_depth={depths[best_i]}; "
      f"deepest gap {train_scores[-1] - val_scores[-1]:+.3f} at max_depth={depths[-1]}")
