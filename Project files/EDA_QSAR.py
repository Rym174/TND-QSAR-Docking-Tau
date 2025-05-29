import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the descriptor data
df = pd.read_csv("Data/ProcessedDescriptors.csv")

# Drop Name and SMILES (non-numeric)
numeric_df = df.drop(columns=["Name", "SMILES"])

# Plot heatmap of correlation
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap of Descriptors")
plt.tight_layout()
plt.savefig("Figures/QSARPlots/correlation_heatmap.png")
plt.show()
