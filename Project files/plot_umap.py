import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv("ProcessedDescriptors.csv")

# Drop Name and SMILES to use only numerical features
features = df.drop(columns=["Name", "SMILES"])

# Run UMAP
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(features)

# Append UMAP columns
df["UMAP1"] = embedding[:, 0]
df["UMAP2"] = embedding[:, 1]

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="Name", s=100, palette="tab20")
plt.title("UMAP Projection of Molecular Descriptors")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save plot
os.makedirs("Figures/QSARPlots", exist_ok=True)
plt.savefig("Figures/QSARPlots/umap_projection.png")
plt.show()
