import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `from src.data import ...` works
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_core_health_prediction_df

warnings.filterwarnings("ignore")
np.random.seed(42)

df = get_core_health_prediction_df()
print(df.head())
X = df.drop("Crop_Health_Label", axis=1)
y = df["Crop_Health_Label"]

# Train-test split 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[3] Split → Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# handle class imbalance 
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"\n[4] scale_pos_weight = {n_neg}/{n_pos} = {scale_pos_weight:.2f}")

# train xgboost
print("\n[5] Training XGBoost...")
 
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42,
    verbosity=0,
)
 
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False,
)

output_path = ROOT / "models" / "xgb_crop_health_model.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(output_path))
print("    Done.")


# evaluate

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
auc         = roc_auc_score(y_test, y_pred_prob)
 
print(f"\n[6] Results:")
print(f"    ROC-AUC: {auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["Unhealthy (0)", "Healthy (1)"]))
print(confusion_matrix(y_test, y_pred))