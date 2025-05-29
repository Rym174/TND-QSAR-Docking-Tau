import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# === Load dataset ===
df = pd.read_csv("ProcessedDescriptors_Reduced.csv")

# === Add docking scores ===
docking_scores = [-8.6, -8.0, -8.3, -8.4, -8.1, -9.1, -8.5, -7.8,
                  -8.7, -8.0, -7.9, -8.3, -8.2, -8.1, -8.8, -8.1]
df["Docking"] = docking_scores

# === Drop non-numeric columns ===
X = df.drop(columns=["Name", "Docking"])
y = df["Docking"]

# === Define models ===
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

results = []

# === Create output folder ===
os.makedirs("QSAR_Performance", exist_ok=True)

# === Cross-validation setup ===
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # Cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Evaluation
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    results.append({"Model": name, "R2": r2, "RMSE": rmse, "MAE": mae})

    # === Plot actual vs predicted ===
    plt.figure()
    plt.scatter(y, y_pred, color='royalblue', edgecolor='black')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel("Actual Docking Score")
    plt.ylabel("Predicted Docking Score")
    plt.title(f"{name} - Actual vs Predicted (CV)")
    plt.tight_layout()
    plt.savefig(f"QSAR_Performance/{name}_cv_actual_vs_predicted.png")
    plt.close()

# === Save comparison table ===
results_df = pd.DataFrame(results)
results_df.to_csv("QSAR_Performance/model_comparison_cv.csv", index=False)

# === Save best model ===
best_model_name = results_df.sort_values("R2", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]
best_model.fit(X, y)  # Re-train on full dataset
joblib.dump(best_model, "BestQSARModel_CV.pkl")

print("\n✅ Cross-validation completed.")
print(f"🏆 Best model based on R²: {best_model_name}")
