import numpy as np
import sys
import pandas as pd
import pickle
from tqdm import tqdm
from inspect import getmro
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Import custom modules
sys.path.append("../../")
from quantum_molecular_encodings.paths import BACKENDS_DIR, EXCEL_DATA_DIR, OVERLAPS_DIR


from quantum_molecular_encodings.matrix import BondOrderMatrix
from quantum_molecular_encodings.encodings.bond import BondFeatureMap
from quantum_molecular_encodings.encodings.overlap import UnitaryOverlap

# Configuration for the quantum encoding layer
ENCODING_LAYER_CONFIG = {
    "n_layers": 1,
    "initial_layer": "ry",
    "entangling_layer": "rxx",
    "n_atom_to_qubit": 1,
    "interleaved": None,
}

# device, date and time for the noise model and backend target
device = "ibm_pittsburgh"
year = 2025
month = 8
day = 10
hour = 18
minute = 59
second = 40

# Generate filename timestamp
timestamp_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}+00:00"
base_filename = f"{device}_{timestamp_str}"

# File paths for noise model and hardware backend target
NOISE_MODEL_PATH = f"{BACKENDS_DIR}/{base_filename}_noise_model.pkl"
TARGET_PATH = f"{BACKENDS_DIR}/{base_filename}_target.pkl"
SAMPLER_TYPE = "statevector"  # Options: "statevector" or "noisy_aer"
NUM_SHOTS = 10_000


def load_pickle(filepath):
    """Loads an object from a pickle file."""
    with open(filepath, "rb") as file:
        return pickle.load(file)


def generate_quantum_circuit(smiles_a, smiles_b, num_atoms_a, num_atoms_b):
    """Generates a transpiled quantum circuit for molecular similarity computation."""
    num_qubits = max(num_atoms_a, num_atoms_b)

    circuit_a = BondFeatureMap(BondOrderMatrix().compute(smiles_a), num_qubits=num_qubits, **ENCODING_LAYER_CONFIG)
    circuit_b = BondFeatureMap(BondOrderMatrix().compute(smiles_b), num_qubits=num_qubits, **ENCODING_LAYER_CONFIG)

    combined_circuit = UnitaryOverlap(circuit_a, circuit_b, barrier=True, measure_all=True).decompose()
    return pass_manager.run(combined_circuit)


# Configure quantum sampler
if SAMPLER_TYPE == "statevector":
    from qiskit.primitives import StatevectorSampler

    sampler = StatevectorSampler(default_shots=NUM_SHOTS)
    device_backend_target = None
    pass_manager = generate_preset_pass_manager(optimization_level=3)

elif SAMPLER_TYPE == "noisy_aer":
    from qiskit_aer.primitives import SamplerV2

    # Load backend and noise model
    device_backend_target = load_pickle(TARGET_PATH)
    noise_model = load_pickle(NOISE_MODEL_PATH)
    pass_manager = generate_preset_pass_manager(optimization_level=3, target=device_backend_target)

    backend_options = {
        "method": "statevector",
        "noise_model": noise_model,
        "shots": NUM_SHOTS,
        "device": "CPU",
        "max_parallel_threads": 0,
        "max_parallel_experiments": 1,
        "max_parallel_shots": 0,
        "statevector_parallel_threshold": 2
    }
    sampler = SamplerV2(default_shots=NUM_SHOTS, options=dict(backend_options=backend_options))
else:
    raise ValueError("Invalid sampler type")

# Load molecular data
DATA_FILE = EXCEL_DATA_DIR / 'hydrocarbon_oxygen_reordered_series.xlsx'

# choose wchich data to use
dataset_type = "alkanes" # options: "alkanes" or "oxygen"
assert dataset_type in ["alkanes", "oxygen"], "Invalid dataset type"

dataframe = pd.read_excel(DATA_FILE, sheet_name='data', header=0)
if dataset_type == "alkanes":
    dataframe = dataframe.loc[dataframe['Number of Oxygens'] == 0]
elif dataset_type == "oxygen":
    dataframe = dataframe.loc[dataframe['Number of Oxygens'] == 1]

smiles_list = dataframe['SMILES'].to_list()
num_carbons = dataframe['Number of Atoms'].to_numpy()

# Prepare log file
log_filename = '_'.join([
    "overlap", dataset_type,
    f"{SAMPLER_TYPE}",
    f"{ENCODING_LAYER_CONFIG['initial_layer']}",
    f"{ENCODING_LAYER_CONFIG['entangling_layer']}",
    f"Lx{ENCODING_LAYER_CONFIG['n_layers']}.txt",
])
log_filename = OVERLAPS_DIR / log_filename
with open(log_filename, 'w') as log_file:
    print(f"Writing into {log_filename}")
    print(f"# primitive_MRO = {getmro(type(sampler))}", file=log_file)
    print(f"# num_shots = {NUM_SHOTS}", file=log_file)
    print(f"# encoding_layer_config = {ENCODING_LAYER_CONFIG}", file=log_file)
    print(f"# noise_model_path = {NOISE_MODEL_PATH}", file=log_file)
    print(f"# backend_target_path = {TARGET_PATH}", file=log_file)
    print(f"# sampler_type = {SAMPLER_TYPE}", file=log_file)
    print("# i, j, expectation-value", file=log_file)


    def run_and_log_expectation(i, j):
        """Runs the quantum circuit and logs the expectation value."""
        circuit = generate_quantum_circuit(smiles_list[i], smiles_list[j], num_carbons[i], num_carbons[j])
        job = sampler.run([circuit])
        result = job.result()[0]
        counts = result.data.meas.get_counts()
        shots = result.metadata['shots']

        # Convert from raw measurement counts to the expectation value
        expectation_value = 0
        for n in range(1, circuit.num_qubits + 1):
            expectation_value += counts.get('0' * n, 0)
        expectation_value /= shots

        print(f"{i}, {j}, {expectation_value}", file=log_file)


    # Load index pairs and process them
    index_pairs = np.genfromtxt(f"delta_num_atoms_{dataset_type}.txt", delimiter=',', dtype=int).T[:2].T
    for idx_a, idx_b in tqdm(index_pairs):
        run_and_log_expectation(idx_a, idx_b)
