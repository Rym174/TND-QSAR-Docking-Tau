import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load your dataset
df = pd.read_csv("ProcessedDescriptors.csv")

# Make sure the folder exists
output_dir = "Figures/QSARPlots"
os.makedirs(output_dir, exist_ok=True)

# Set aesthetic style
sns.set(style="whitegrid")

# List of descriptor columns (excluding Name and SMILES)
descriptor_cols = [col for col in df.columns if col not in ["Name", "SMILES"]]

# Plot boxplot and KDE for each descriptor
for col in descriptor_cols:
    plt.figure(figsize=(10, 5))

    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(f"Boxplot of {col}")

    # KDE plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df[col], fill=True, color="salmon")
    plt.title(f"KDE Plot of {col}")

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{col}_box_kde.png")
    plt.close()

print("✅ All boxplots and KDE plots saved to Figures/QSARPlots/")
