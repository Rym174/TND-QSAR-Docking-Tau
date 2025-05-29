import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load the descriptor data (same folder)
df = pd.read_csv("ProcessedDescriptors.csv")

# Drop non-numeric columns
numeric_df = df.drop(columns=["Name", "SMILES"])

# Standardize the descriptors
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Apply t-SNE with fast settings
tsne = TSNE(n_components=2, random_state=42, perplexity=5, init='pca', n_iter=500)
tsne_result = tsne.fit_transform(scaled_data)

# Add compound names back
tsne_df = pd.DataFrame(tsne_result, columns=["tSNE1", "tSNE2"])
tsne_df["Compound"] = df["Name"]

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=tsne_df, x="tSNE1", y="tSNE2", hue="Compound", s=100)
plt.title("t-SNE Clustering of Triazole–Naphthalene Derivatives")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Save the plot to current folder
plt.savefig("tSNE_plot.png")
plt.show()
