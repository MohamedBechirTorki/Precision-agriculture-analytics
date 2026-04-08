import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_environmental_stress_analysis_df

warnings.filterwarnings("ignore")
np.random.seed(42)

# Load 
df = get_environmental_stress_analysis_df()
print(f"Loaded: {df.shape}")

FEATURES = ["Pest_Hotspots", "Pest_Damage", "Thermal_Images",
            "Soil_Moisture", "Rainfall", "Water_Flow",
            "Drainage_Features", "Organic_Matter"]
TARGET = "Crop_Stress_Indicator"

X = df[FEATURES].copy()
y = df[TARGET]

# Non-linear feature engineering 

X["Pest_Hotspots_sq"]    = X["Pest_Hotspots"] ** 2
X["Pest_Damage_sq"]      = X["Pest_Damage"] ** 2
X["Soil_Moisture_sq"]    = X["Soil_Moisture"] ** 2
X["Pest_Hotspots_sqrt"]  = np.sqrt(X["Pest_Hotspots"])
X["Pest_Damage_sqrt"]    = np.sqrt(X["Pest_Damage"])
X["Rainfall_log"]        = np.log1p(X["Rainfall"])
X["Soil_Moisture_log"]   = np.log1p(X["Soil_Moisture"])

X["Pest_x_Damage"]       = X["Pest_Hotspots"] * X["Pest_Damage"]
X["Pest_x_Thermal"]      = X["Pest_Hotspots"] * X["Thermal_Images"]
X["Drought_stress"]      = (1 - X["Soil_Moisture"] / 95) * (1 - X["Rainfall"] / 400)
X["Pest_x_Drought"]      = X["Pest_Hotspots"] * X["Drought_stress"]
X["Damage_x_Drought"]    = X["Pest_Damage"] * X["Drought_stress"]
X["Combined_stress"]     = X["Pest_Hotspots"] * X["Pest_Damage"] * X["Drought_stress"]

X["Pest_to_Water"]       = X["Pest_Hotspots"] / (X["Water_Flow"] + 1)
X["Damage_to_Organic"]   = X["Pest_Damage"] / (X["Organic_Matter"] + 0.1)

print(f"    Features: {len(X.columns)} (was {len(FEATURES)})")

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nSplit → Train: {len(X_train)} | Test: {len(X_test)}")

# Transform target to normal distribution
qt = QuantileTransformer(output_distribution="normal", random_state=42)
y_train_t = qt.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_t  = qt.transform(y_test.values.reshape(-1, 1)).ravel()

# Model
print("\nTraining XGBoost Regressor...")
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    reg_alpha=0.05,
    reg_lambda=0.5,
    gamma=0.05,
    eval_metric="rmse",
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0,
)

model.fit(
    X_train, y_train_t,
    eval_set=[(X_test, y_test_t)],
    verbose=False,
)

# Predict & inverse transform
y_pred = qt.inverse_transform(
    model.predict(X_test).reshape(-1, 1)
).ravel().clip(0, 99)

# Save
output_path = ROOT / "models" / "xgb_environmental_stress_model.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(output_path))
print(f"    Saved → {output_path}")

# Evaluate
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nResults:")
print(f"    MAE  : {mae:.4f}")
print(f"    RMSE : {rmse:.4f}")
print(f"    R²   : {r2:.4f}")

# Feature importance (top 15)
print(f"\nTop 15 Feature Importances:")
importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.nlargest(15).round(4))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sample = np.random.choice(len(y_test), size=min(3000, len(y_test)), replace=False)
y_test_arr = y_test.values

# Actual vs Predicted — should show concave curve
axes[0].scatter(y_test_arr[sample], y_pred[sample], alpha=0.3, s=10, color="steelblue")
axes[0].plot([0, 100], [0, 100], "r--", linewidth=2, label="Perfect")
axes[0].set_xlabel("Actual Stress")
axes[0].set_ylabel("Predicted Stress")
axes[0].set_title(f"Actual vs Predicted (R²={r2:.3f})")
axes[0].legend()

# Residuals
residuals = y_test_arr - y_pred
axes[1].scatter(y_pred[sample], residuals[sample], alpha=0.3, s=10, color="coral")
axes[1].axhline(0, color="black", linewidth=1.5, linestyle="--")
axes[1].set_xlabel("Predicted Stress")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals (random = good)")

# Top feature importances
importance.nlargest(15).sort_values().plot(kind="barh", ax=axes[2], color="steelblue")
axes[2].set_title("Top 15 Feature Importances")
axes[2].set_xlabel("Score")

plt.tight_layout()
plot_path = ROOT / "models" / "environmental_stress_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n    Plot saved → {plot_path}")
plt.show()