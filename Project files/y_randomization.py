import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import random

# === Load your descriptor file ===
df = pd.read_csv("ProcessedDescriptors_Reduced.csv")

# === Add docking scores ===
docking_scores = [-8.6, -8.0, -8.3, -8.4, -8.1, -9.1, -8.5, -7.8,
                  -8.7, -8.0, -7.9, -8.3, -8.2, -8.1, -8.8, -8.1]
df["Docking"] = docking_scores

# === Prepare features and target ===
X = df.select_dtypes(include=[np.number]).drop(columns=["Docking"])
y = df["Docking"]

# === Create output directory ===
os.makedirs("QSAR_Performance", exist_ok=True)

# === Run Y-randomization ===
r2_scores = []
n_runs = 50
for i in range(n_runs):
    y_shuffled = y.sample(frac=1.0, random_state=i).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y_shuffled, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# === Plot and save ===
plt.figure(figsize=(7, 5))
plt.hist(r2_scores, bins=10, color="skyblue", edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--", label="R² = 0")
plt.title("Y-Randomization: R² Score Distribution")
plt.xlabel("R² Score (Shuffled)")
plt.ylabel("Frequency")
plt.legend()

# === Save plot ===
plt.savefig("QSAR_Performance/y_randomization_r2_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
