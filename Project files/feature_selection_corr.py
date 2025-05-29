import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("ProcessedDescriptors.csv")

# Drop non-numeric columns
numeric_df = df.drop(columns=["Name", "SMILES"])

# Compute correlation matrix
corr_matrix = numeric_df.corr().abs()

# Create mask for upper triangle
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation > 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print("Highly correlated descriptors to drop:", to_drop)

# Drop those features
reduced_df = numeric_df.drop(columns=to_drop)

# Save reduced descriptor dataset
reduced_df.insert(0, "Name", df["Name"])
reduced_df.insert(1, "SMILES", df["SMILES"])
reduced_df.to_csv("ProcessedDescriptors_Reduced.csv", index=False)

# Optional: visualize reduced correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(reduced_df.drop(columns=["Name", "SMILES"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap After Filtering")
plt.tight_layout()
plt.savefig("correlation_reduced_heatmap.png")
plt.show()
