import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from quantum_molecular_encodings.paths import MPL_FILE, EXCEL_DATA_DIR

# Load molecular data
HYDROCARBON_DATA_FILE = EXCEL_DATA_DIR / 'hydrocarbon_oxygen_reordered_series.xlsx'
dataframe = pd.read_excel(HYDROCARBON_DATA_FILE, sheet_name='data', header=0)
dataframe = dataframe.loc[dataframe['Number of Oxygens'] == 1]

smiles = dataframe['SMILES'].to_list()
num_carbons = dataframe['Number of Atoms'].to_numpy()

# Construct a matrix
ij = np.tril_indices(len(smiles))
num_atoms_mask = np.empty((len(smiles), len(smiles)))
num_atoms_mask[:] = np.nan

valid_index_pairs = open("delta_num_atoms_oxygen.txt", 'w')
print("# i, j, num-i, num-j, Delta-num-atoms", file=valid_index_pairs)

for i, j in zip(*ij):
    delta_num_atoms = np.abs(num_carbons[i] - num_carbons[j])
    num_atoms_mask[i, j] = delta_num_atoms
    print(f"{i}, {j}, {num_carbons[i]}, {num_carbons[j]}, {delta_num_atoms}", file=valid_index_pairs)

valid_index_pairs.close()

plt.style.use(MPL_FILE)
fig, ax = plt.subplots(figsize=(3.2 * 1.5, 3.2 * 1.5), constrained_layout=True)
ax.set_xticks(np.arange(len(smiles)))
ax.set_xticklabels(smiles, rotation=90, fontsize='x-small')
ax.set_yticks(np.arange(len(smiles)))
ax.set_yticklabels(smiles, fontsize='x-small')
ax.tick_params(axis='both', direction='out')
im = ax.imshow(num_atoms_mask, cmap="magma_r", vmin=-0.5, vmax=10.5)

cbaxes = inset_axes(ax, width="65%", height="4%", loc="upper right")
plt.colorbar(im, cax=cbaxes, orientation='horizontal', label=r"$\Delta$(number of atoms)")
cbaxes.xaxis.set_ticks(np.arange(11))
cbaxes.xaxis.set_ticks_position('bottom')
cbaxes.xaxis.set_label_position('bottom')
plt.show()
