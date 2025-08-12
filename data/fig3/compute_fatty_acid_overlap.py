import sys
sys.path.append("../../")


import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs

from qiskit.transpiler.exceptions import TranspilerError
from qiskit_aer.primitives import SamplerV2 as SamplerAer

from quantum_molecular_encodings.paths import FATTY_ACID_EXCEL_DIR, FATTY_ACID_OVERLAPS_DIR, FIGURES_DIR
from quantum_molecular_encodings.supporting_functions import generate_overlap_circuit, molecule_to_pdf
from quantum_molecular_encodings.processing import bootstrap_smiles_list
from quantum_molecular_encodings.processing import get_unique_smiles, reverse_smiles
from quantum_molecular_encodings.processing import count_atoms_and_check_validity, contains_cooh_group



# load LMSD
LMSD_FILEPATH = f"{FATTY_ACID_EXCEL_DIR}/LMSD.xlsx"
df = pd.read_excel(LMSD_FILEPATH, engine='openpyxl')
df['smiles_rdkit'] = df['smiles'].apply(get_unique_smiles)
df['smiles_reordered'] = df['smiles_rdkit'].apply(reverse_smiles)


# filter data
has_triple_bond = False
has_isotopes = False
num_double_bonds = 2

# Filter the DataFrame to only include entries with two oxygen atoms, valid atoms (C, H, O), no triple bonds, no isotopes, and the specified number of double bonds
df_filtered = df[df['smiles_rdkit'].apply(lambda x: (
    count_atoms_and_check_validity(x, has_triple_bond, has_isotopes, num_double_bonds)[1] == 2 and
    count_atoms_and_check_validity(x, has_triple_bond, has_isotopes, num_double_bonds)[2] and
    not count_atoms_and_check_validity(x, has_triple_bond, has_isotopes, num_double_bonds)[3] and
    not count_atoms_and_check_validity(x, has_triple_bond, has_isotopes, num_double_bonds)[4]
#    count_atoms_and_check_validity(x, has_triple_bond, has_isotopes, num_double_bonds)[5]   ### Uncomment to filter by number of double bonds
))]

# Filter the DataFrame to only include entries with a COOH group (carboxylic acid)
df_filtered_by_cooh = df_filtered[df_filtered['smiles_rdkit'].apply(contains_cooh_group)]











IMAGE_SAVEPATH = f"{FIGURES_DIR}/fig8/" # figure 8 of the paper 




# Array of carbon numbers to filter by
c_array = [34]  # chain contraction for fatty acids with 34 carbons.

feature_map_config = {
    "n_layers": 1,
    "initial_layer": "ry",
    "entangling_layer": "rxx",
    "n_atom_to_qubit": 1,
    "interleaved": None,
}

sampler = SamplerAer()

# Initialize summary table
summary_table = []

# Iterate over each carbon number in c_array
for c_num in c_array:
    print(f"Processing carbon number: {c_num}")
    OVERLAPS_DIR = FATTY_ACID_OVERLAPS_DIR / f"carbon_{c_num}"
    os.makedirs(OVERLAPS_DIR, exist_ok=True)
    # Filter the DataFrame to only include entries with a COOH group (carboxylic acid)
    df_filtered_by_carbon = df_filtered[df_filtered['smiles_rdkit'].apply(contains_cooh_group)]

    # Filter the DataFrame to only include entries with c_num number of carbon atoms
    df_filtered_by_cooh = df_filtered_by_carbon[df_filtered_by_carbon['smiles_rdkit'].apply(lambda x: count_atoms_and_check_validity(x, has_triple_bond, has_isotopes, num_double_bonds)[0] == c_num)]

    # Create directory for the current carbon number
    EXCEL_FILEPATH = os.path.join(FATTY_ACID_EXCEL_DIR, f"fattyacids_carbon_{c_num}.xlsx")
    
    
    # Save the filtered DataFrame to a CSV file in the directory
    df_filtered_by_cooh.to_excel(EXCEL_FILEPATH, index=False)
    
    # Save molecular structures as images in the directory
    for index, row in df_filtered_by_cooh.iterrows():
        mol = Chem.MolFromSmiles(row['smiles_rdkit'])
        if mol:
            molecule_to_pdf(mol, row['LM_ID'], IMAGE_SAVEPATH)
    
    # Append summary information for the current carbon number
    summary_table.append((c_num, len(df_filtered_by_cooh)))

    smiles_list = df_filtered_by_cooh['smiles_reordered'].to_list()

    # add missing stereochemistries if any
    smiles_list = bootstrap_smiles_list(smiles_list)

    molecules = [Chem.MolFromSmiles(smile) for smile in smiles_list]

    # Generate fingerprints for each molecule
    fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)

    #fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4, includeChirality=True, fpSize=2048)
    fingerprints = ([fpgen.GetFingerprint(mol) for mol in molecules])
    

    # Calculate Tanimoto similarity matrix
    n = len(fingerprints)
    tanimoto_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(0,i+1):
            if i != j:
                tanimoto_matrix[i][j] = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[j])
    
            else:
                tanimoto_matrix[i][j] = 1.0  # Similarity with itself is 1

    # Display the Tanimoto similarity matrix

    tanimoto_df = pd.DataFrame(tanimoto_matrix, index=smiles_list, columns=smiles_list)
    tanimoto_df.to_excel(os.path.join(OVERLAPS_DIR, "tanimoto.xlsx"), index=True)

    # quantum overlaps

    quantum_overlap = np.zeros((n, n))
    circuit_size = np.zeros((n, n))
    
    for i in range(n):
        for j in range(0,i+1):
            if i != j:
                try:
                    circuit = generate_overlap_circuit(smiles_list[i], smiles_list[j], c_num+10, c_num+10, feature_map_config)
                except TranspilerError as e:
                    print(circuit)
                    print(f"n = {n}, i = {i}, j = {j}")
                    print(f"smiles_list[i] = {smiles_list[i]}, smiles_list[j] = {smiles_list[j]}")
                    print(f"Error: {e}")
                    continue
                num_qubits = circuit.num_qubits
                job = sampler.run([circuit])
                
                result = job.result()[0]
                try:
                    counts = result.data.meas.get_counts()
                except ValueError:
                    print(f"n = {n}, i = {i}, j = {j}")
                    print(f"smiles_list[i] = {smiles_list[i]}, smiles_list[j] = {smiles_list[j]}")
                    print(f"result = {result}")
                    print(f"result.data = {result.data}")
                    print(f"result.data.meas = {result.data.meas}")
                    
                _shots = result.metadata['shots']
                expectation_value = counts.get('0' * circuit.num_qubits, 0)

                expectation_value /= _shots
                quantum_overlap[i][j] = expectation_value
                circuit_size[i][j] = num_qubits
            else:
                quantum_overlap[i][j] = 1.0  # Overlap with itself is 1
                circuit_size[i][j] = 0
            

    # Display the quantum overlap matrix
    quantum_overlap_df = pd.DataFrame(quantum_overlap, index=smiles_list, columns=smiles_list)
    quantum_overlap_df.to_excel(os.path.join(OVERLAPS_DIR, f"statevector_{feature_map_config['initial_layer']}_{feature_map_config['entangling_layer']}_Lx{feature_map_config['n_layers']}.xlsx"), index=True)
    circuit_size_df = pd.DataFrame(circuit_size, index=smiles_list, columns=smiles_list)
    circuit_size_df.to_excel(os.path.join(OVERLAPS_DIR, f"circuit_size_{feature_map_config['initial_layer']}_{feature_map_config['entangling_layer']}_Lx{feature_map_config['n_layers']}.xlsx"), index=True)


# Print summary table
print(f"Summary Table for unsaturation: {num_double_bonds-1}")
print("Carbon Number | Number of Molecules")
print("-------------------------------")
for c_num, count in summary_table:
    print(f"{c_num:<13} | {count}")

print("DataFrames and images have been saved successfully.")


