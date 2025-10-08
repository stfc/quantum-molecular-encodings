"""Same as fig4.py but for noisy aer. Code duplication to make it easy to follow the publication."""

import sys
sys.path.append("../")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from quantum_molecular_encodings.paths import MPL_FILE, EXCEL_DATA_DIR, OVERLAPS_DIR
SAMPLER_TYPE = "noisy_aer"
expectation_label = "Fidelity"
# Configure the plot style
plt.style.use(MPL_FILE)

# Create a heatmap visualization of the matrix
fig, axes = plt.subplots(
    2, 3,
    figsize=(3.2 * 2, 3.2 * 1.6),
    gridspec_kw=dict(wspace=0.05, hspace=0.05),
    # constrained_layout=True,
)

for ax in axes.flat:
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

for ax, gate in zip(axes.flat[:3], ["rxx", "ryy", "rzz"]):

    # Configuration for the quantum encoding layer
    ENCODING_LAYER_CONFIG = {
        "n_layers": 1,
        "initial_layer": "ry",
        "entangling_layer": gate,
        "n_atom_to_qubit": 1,
        "interleaved": None,
    }

    # Generate filename dynamically
    log_filename = "_".join([
        "overlap", "alkanes",
        f"{SAMPLER_TYPE}",
        f"{ENCODING_LAYER_CONFIG['initial_layer']}",
        f"{ENCODING_LAYER_CONFIG['entangling_layer']}",
        f"Lx{ENCODING_LAYER_CONFIG['n_layers']}.txt"
    ])

    # Load hydrocarbon data from an Excel file
    DATAFILE = EXCEL_DATA_DIR / "hydrocarbon_oxygen_series_unique_smiles.xlsx"
    dataframe = pd.read_excel(DATAFILE, sheet_name='Sheet1', header=0)
    dataframe = dataframe.loc[dataframe['Number of Oxygens'] == 0]

    smiles_list = dataframe['Unique_SMILES'].to_list()  # List of SMILES strings representing molecules
    num_carbons_array = dataframe['Number of Atoms'].to_numpy()  # Array of carbon counts for molecules

    # Initialize a lower triangular matrix for storing expectation values
    matrix_size = len(smiles_list)
    indices_lower_triangular = np.tril_indices(matrix_size)
    expectation_matrix = np.full((matrix_size, matrix_size), np.nan)  # Initialize with NaN values


    # Load expectation values from the log file
    data = np.genfromtxt(
        f"{OVERLAPS_DIR}/" + log_filename,
        delimiter=',',
        dtype=[('row', int), ('col', int), ('value', float)]
    )

    # Populate the lower triangular matrix with the expectation values
    for i, j, expectation_value in data:
        expectation_matrix[i, j] = expectation_value

    ax.set_title(fr"${gate[0].upper()}_{{{gate[1:]}}}$", fontsize="xx-large")
    im = ax.imshow(expectation_matrix, cmap="magma_r")  # Heatmap of the matrix

    # Add a histogram of expectation values as an inset
    pdfaxes = inset_axes(
        ax, width="35%", height="35%", loc="upper right",
    )
    pdfaxes.hist(expectation_matrix.flat, bins=20)  # Flatten the matrix and bin the values
    pdfaxes.set_xlabel(expectation_label)
    pdfaxes.set_ylabel("Frequency")

for ax, gate in zip(axes.flat[3:], ["rxx", "ryy", "rzz"]):
    # Configuration for the quantum encoding layer
    ENCODING_LAYER_CONFIG = {
        "n_layers": 1,
        "initial_layer": "ry",
        "entangling_layer": gate,
        "n_atom_to_qubit": 1,
        "interleaved": None,
    }

    # Generate filename dynamically
    log_filename = "_".join([
        "overlap", "oxygen",
        f"{SAMPLER_TYPE}",
        f"{ENCODING_LAYER_CONFIG['initial_layer']}",
        f"{ENCODING_LAYER_CONFIG['entangling_layer']}",
        f"Lx{ENCODING_LAYER_CONFIG['n_layers']}.txt"
    ])
    DATAFILE = EXCEL_DATA_DIR / "hydrocarbon_oxygen_reordered_series.xlsx"
    dataframe = pd.read_excel(DATAFILE, sheet_name='data', header=0)
    dataframe = dataframe.loc[dataframe['Number of Oxygens'] == 1]

    smiles_list = dataframe['SMILES'].to_list()  # List of SMILES strings representing molecules
    num_carbons_array = dataframe['Number of Atoms'].to_numpy()  # Array of carbon counts for molecules

    # Initialize a lower triangular matrix for storing expectation values
    matrix_size = len(smiles_list)
    indices_lower_triangular = np.tril_indices(matrix_size)
    expectation_matrix = np.full((matrix_size, matrix_size), np.nan)  # Initialize with NaN values

    # Load expectation values from the log file
    data = np.genfromtxt(
        f"{OVERLAPS_DIR}/" + log_filename,
        delimiter=',',
        dtype=[('row', int), ('col', int), ('value', float)]
    )

    # Populate the lower triangular matrix with the expectation values
    for i, j, expectation_value in data:
        expectation_matrix[i, j] = expectation_value

    im = ax.imshow(expectation_matrix, cmap="magma_r")  # Heatmap of the matrix

    # Add a histogram of expectation values as an inset
    pdfaxes = inset_axes(
        ax, width="35%", height="35%", loc="upper right",
    )
    pdfaxes.hist(expectation_matrix.flat, bins=20)  # Flatten the matrix and bin the values
    pdfaxes.set_xlabel(expectation_label)
    pdfaxes.set_ylabel("Frequency")

# Add a horizontal colorbar to represent expectation values
cbaxes = inset_axes(
    axes[1, 1], width="100%", height="7%", loc="lower center",
    bbox_to_anchor=(0, -0.15, 1, 1), bbox_transform=axes[1, 1].transAxes
)
plt.colorbar(im, cax=cbaxes, orientation='horizontal', label=expectation_label)
cbaxes.xaxis.set_ticks_position('bottom')
cbaxes.xaxis.set_label_position('bottom')

plt.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.12)

plt.savefig("fig10.pdf")
plt.close()


