import os

# === Configuration ===
vina_path = r"C:\Users\poula\Documents\The Scripps Research Institute\Vina\vina.exe"
receptor = "6hrf.pdbqt"
ligands = ["TND-11.pdbqt", "TND-12.pdbqt", "TND-13.pdbqt", "TND-14.pdbqt", "TND-15.pdbqt"]

# === Grid box coordinates ===
center_x = 139.299
center_y = 161.886
center_z = 197.094
size_x = 90
size_y = 126
size_z = 84

# === Output directory (inside the current folder) ===
output_dir = "vina_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Dock each ligand ===
for ligand in ligands:
    name = ligand.split(".")[0]
    out_pdbqt = os.path.join(output_dir, f"{name}_out.pdbqt")
    log_file = os.path.join(output_dir, f"{name}_log.txt")

    command = f'"{vina_path}" --receptor {receptor} --ligand {ligand} '
    command += f'--center_x {center_x} --center_y {center_y} --center_z {center_z} '
    command += f'--size_x {size_x} --size_y {size_y} --size_z {size_z} '
    command += f'--out "{out_pdbqt}" --log "{log_file}"'

    print(f"\n⏳ Docking {ligand}...\n{command}\n")
    os.system(command)

print("\n✅ Docking completed for all ligands.")
