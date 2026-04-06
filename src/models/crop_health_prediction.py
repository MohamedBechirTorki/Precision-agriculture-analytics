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
print(X.head())
print(y.head())