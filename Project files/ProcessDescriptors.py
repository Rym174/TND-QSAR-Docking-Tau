import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

# Make sure the Data directory exists
os.makedirs("Data", exist_ok=True)

# List of SMILES
smiles_list = [
    {"Name": "TND", "SMILES": "O=C1C2=CC=CC3=C2C(=CC=C3)C4=NN=C([NH]4)C5=CC=CC6=C5C1=CC=C6"},
    {"Name": "TND-1", "SMILES": "COC1=CC2=C3C(=CC=C2)C4=NN=C([NH]4)C5=CC=CC6=C5C(=CC(=C6)OC)C(=O)C3=C1"},
    {"Name": "TND-2", "SMILES": "CC1=CC2=C3C(=C1)C(=O)C4=CC(=CC5=C4C(=CC=C5)C6=NN=C([NH]6)C3=CC=C2)CO"},
    {"Name": "TND-3", "SMILES": "COC1=CC2=C3C(=CC(=C2)C)C(=O)C4=CC=CC5=C4C(=CC=C5)C6=NN=C([NH]6)C3=C1"},
    {"Name": "TND-4", "SMILES": "COC1=CC2=C3C(=CC=C2)C(=O)C4=CC=CC5=C4C(=CC(=C5)OC)C6=NN=C([NH]6)C3=C1"},
    {"Name": "TND-5", "SMILES": "CC1=CC2=C3C(=CC=C2)C(=O)C4=CC=CC5=C4C(=CC(=C5)CO)C6=NN=C([NH]6)C3=C1"},
    {"Name": "TND-6", "SMILES": "C[N]1C2=NN=C1C3=CC(=CC4=C3C(=CC=C4)C(=O)C5=CC=CC6=C5C2=CC(=C6)C)CO"},
    {"Name": "TND-7", "SMILES": "O=C1c2cccc3cc(C)c(CO)c(c4nnc([NH]4)c4cccc5cccc1c45)c23"},
    {"Name": "TND-8", "SMILES": "CC1=CC2=C3C(=CC=C2)C(=O)C4=CC=CC5=C4C(=CC(=C5)OC6=CC=CC=C6)C7=NN=C([NH]7)C3=C1"},
    {"Name": "TND-9", "SMILES": "CCOC1=CC2=C3C(=CC=C2)C(=O)C4=CC=CC5=C4C(=CC(=C5)CC)C6=NN=C([NH]6)C3=C1"},
    {"Name": "TND-10", "SMILES": "CCOC1=CC2=C3C(=CC=C2)C(=O)C4=CC=CC5=C4C(=CC(=C5)OC)C6=NN=C([NH]6)C3=C1"},
    {"Name": "TND-11", "SMILES": "O=C1c2cccc3cccc(c4nnc(n4N)c4cccc5cccc1c45)c23"},
    {"Name": "TND-12", "SMILES": "O=C1c2cccc3cccc(c4[NH]c(n[n+]4N)c4cccc5cccc1c45)c23"},
    {"Name": "TND-13", "SMILES": "On1c2nnc1c1cccc3cccc(C(=O)c4cccc5cccc2c45)c13"},
    {"Name": "TND-14", "SMILES": "O=C1c2cccc3cccc(c4nnc(n4F)c4cccc5cccc1c45)c23"},
    {"Name": "TND-15", "SMILES": "O=C1c2cccc3cccc(c4nnc(n4OC)c4cccc5cccc1c45)c23"}
]

# Convert SMILES to RDKit molecules
for compound in smiles_list:
    compound["Mol"] = Chem.MolFromSmiles(compound["SMILES"])

# Calculate descriptors
for compound in smiles_list:
    mol = compound["Mol"]
    if mol:
        compound["MolWt"] = Descriptors.MolWt(mol)
        compound["LogP"] = Descriptors.MolLogP(mol)
        compound["TPSA"] = Descriptors.TPSA(mol)
        compound["NumHDonors"] = Descriptors.NumHDonors(mol)
        compound["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
        compound["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
        compound["RingCount"] = Descriptors.RingCount(mol)

# Create DataFrame and save to CSV
df = pd.DataFrame(smiles_list)
df.drop(columns=["Mol"], inplace=True)
df.to_csv("Data/ProcessedDescriptors.csv", index=False)
