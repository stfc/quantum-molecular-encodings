import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

feature_map_config = {
    "n_layers": 1,
    "initial_layer": "ry",
    "entangling_layer": "rxx",
    "n_atom_to_qubit": 1,
    "interleaved": None,
}
c_num = 34

os.makedirs(f"fattyacids/carbon_{c_num}", exist_ok=True)
PLOTDIR = f"fattyacids/carbon_{c_num}/"

# Define the path to the directory containing the Excel files
directory = f"../data/fattyacids/carbon_{c_num}"

# Load the Tanimoto similarity matrix
tanimoto_df = pd.read_excel(os.path.join(directory, "tanimoto.xlsx"), index_col=0)

# Load the quantum overlap matrix
quantum_overlap_df = pd.read_excel(os.path.join(directory, f"statevector_{feature_map_config['initial_layer']}_{feature_map_config['entangling_layer']}_Lx{feature_map_config['n_layers']}.xlsx"), index_col=0)

# Load the circuit size matrix
circuit_size_df = pd.read_excel(os.path.join(directory, f"circuit_size_{feature_map_config['initial_layer']}_{feature_map_config['entangling_layer']}_Lx{feature_map_config['n_layers']}.xlsx"), index_col=0)

# Configure the plot style
plt.style.use("../molecular_encodings/molecular.mplstyle")

# Function to plot heatmap with inset colorbar and save to file

fig, axes = plt.subplots(
    1, 2,
    figsize=(3.2 * 1.6 * 0.7, 2.5 * 0.7),
    layout='constrained'
)


# Plot and save Tanimoto similarity matrix
df, title, cmap, ax = tanimoto_df, "Tanimoto similarity", "magma_r", axes[0]

matrix_size = len(df.index)
expectation_matrix = np.full((matrix_size, matrix_size), np.nan)  # Initialize with NaN values
indices_lower_triangular = np.tril_indices(matrix_size)

for i, j in zip(*indices_lower_triangular):
    expectation_matrix[i, j] = df.iloc[i, j]

im = ax.imshow(expectation_matrix, cmap=cmap, vmin=0, vmax=1)

ax.set_xticks(range(len(df.index)))
ax.set_xticklabels([f"FA{i:d}" for i in range(1, len(df.columns) + 1)], fontsize='small', rotation=90, weight='bold')
ax.set_yticks(range(len(df.index)))
ax.set_yticklabels([f"FA{i:d}" for i in range(1, len(df.columns) + 1)], fontsize='small', weight='bold')
ax.tick_params(axis='both', direction='out')
ax.set_title(title)

# Plot and save Quantum Overlap matrix
df, title, cmap, ax = quantum_overlap_df, "QME overlap", "magma_r", axes[1]

matrix_size = len(df.index)
expectation_matrix = np.full((matrix_size, matrix_size), np.nan)  # Initialize with NaN values
indices_lower_triangular = np.tril_indices(matrix_size)

for i, j in zip(*indices_lower_triangular):
    expectation_matrix[i, j] = df.iloc[i, j]

    ax.text(j, i, f"{circuit_size_df.iloc[i, j]:d}", ha='center', va='center',
            color='w' if df.iloc[i, j] > 0.5 else 'k', fontsize='medium')

im = ax.imshow(expectation_matrix, cmap=cmap, vmin=0, vmax=1)

ax.set_xticks(range(len(df.index)))
ax.set_xticklabels([f"FA{i:d}" for i in range(1, len(df.columns) + 1)], fontsize='small', rotation=90, weight='bold')
ax.set_yticks(range(len(df.index)))
ax.set_yticklabels([f"FA{i:d}" for i in range(1, len(df.columns) + 1)], fontsize='small', weight='bold')
ax.tick_params(axis='both', direction='out')
ax.set_title(title)

plt.colorbar(im, ax=axes[1], location='right', orientation='vertical')

plt.savefig("fig3.pdf")
plt.show()