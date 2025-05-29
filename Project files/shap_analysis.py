import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
import os

# === Load processed descriptors (Reduced version) ===
df = pd.read_csv("ProcessedDescriptors_Reduced.csv")

# === Add docking scores for all 16 compounds ===
docking_scores = [
    -8.6, -8.0, -8.3, -8.4, -8.1, -9.1, -8.5, -7.8,
    -8.7, -8.0, -7.9, -8.3, -8.2, -8.1, -8.8, -8.1
]
df["Docking"] = docking_scores

# === Define input features (X) and target (y) ===
X = df.drop(columns=["Name", "SMILES", "Docking"])
y = df["Docking"]

# === Train the Random Forest model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# === SHAP analysis ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# === Create output folder ===
os.makedirs("Figures/FeatureImportance", exist_ok=True)

# === SHAP bar plot ===
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig("Figures/FeatureImportance/shap_summary_bar.png")

# === SHAP beeswarm plot ===
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("Figures/FeatureImportance/shap_summary_beeswarm.png")

print("✅ SHAP analysis completed and plots saved.")
