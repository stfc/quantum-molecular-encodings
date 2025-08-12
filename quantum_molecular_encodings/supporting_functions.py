import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import cairosvg

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryOverlap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator




from quantum_molecular_encodings.matrix import BondOrderMatrix
from quantum_molecular_encodings.encodings.bond import BondFeatureMap
from quantum_molecular_encodings.encodings.overlap import UnitaryOverlap

def coulomb_matrix(smiles: str, add_hydrogens: bool = False, bond_coupling: float = 1.0) -> np.ndarray:
    """
    Computes the adjacent Coulomb matrix for a given molecule specified by a SMILES string,
    using specific average bond lengths for adjacent atom pairs.
    
    Parameters:
    - smiles (str): The SMILES string representing the molecule.
    - add_hydrogens (bool): Whether to add hydrogen atoms to the molecule.
    
    Returns:
    - np.ndarray: The Coulomb matrix of the molecule.
    """
    # Load the molecule from the SMILES string
    molecule = Chem.MolFromSmiles(smiles)
    
    # Add hydrogen atoms if specified
    if add_hydrogens == True:
        molecule = Chem.AddHs(molecule)
    
    # Get the atomic numbers of the atoms
    atomic_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    
    # Number of atoms
    num_atoms = len(atomic_numbers)
    
    # Initialize the Coulomb matrix
    coulomb_matrix = np.zeros((num_atoms, num_atoms))
    
    # Fill in the Coulomb matrix
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                # Diagonal elements: 0.5 * Z_i^2.4
                coulomb_matrix[i, j] = (0.5 * atomic_numbers[i] ** 2.4) 
                # coulomb_matrix[i, j] = (atomic_numbers[i]) 
            else:
                # Find the bond between atoms i and j
                bond = molecule.GetBondBetweenAtoms(i, j)
                if bond:
                    bond_type = bond.GetBondType()
                    if bond_type == Chem.rdchem.BondType.SINGLE:
                        distance = 1
                    elif bond_type == Chem.rdchem.BondType.DOUBLE:
                        distance = 2
                    elif bond_type == Chem.rdchem.BondType.TRIPLE:
                        distance = 3
                    elif bond_type == Chem.rdchem.BondType.AROMATIC:
                        distance = 1.5
                    coulomb_matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j] / distance) * bond_coupling
    
    return coulomb_matrix

def matrix_to_circuit(matrix, num_qubits, n_layers: int = 1, reverse_bits: bool = False, initial_layer: str = 'rx', entangling_layer: str = 'rzz', n_atom_to_qubit: int = 1, interleaved: str = None) -> QuantumCircuit:
    """
    Converts a matrix to a QuantumCircuit object.
    
    Parameters:
    - matrix (np.ndarray): The matrix to convert.
    
    Returns:
    - QuantumCircuit: The QuantumCircuit object representing the matrix.
    """
    # Get the number of qubits required to represent the matrix
    matrix_size = matrix.shape[0]

    # Toggle reverse bits
    if reverse_bits == True:
        m = np.flip(np.arange(num_qubits - matrix_size * n_atom_to_qubit, num_qubits))
    else:
        m = np.arange(0, matrix_size * n_atom_to_qubit)
    
    m = np.reshape(m, (matrix_size, n_atom_to_qubit))
    
    # Initialize the QuantumCircuit object
    qc = QuantumCircuit(num_qubits)
    for _ in range(n_layers):
        for i in range(matrix_size):
            if initial_layer == 'ry':
                for k in range(n_atom_to_qubit):
                    qc.ry(matrix[i, i], m[i,k])
            elif initial_layer == 'rz':
                for k in range(n_atom_to_qubit):
                    qc.rz(matrix[i, i], m[i,k])
            else:
                for k in range(n_atom_to_qubit):
                    qc.rx(matrix[i, i], m[i,k])
        if interleaved == 'cnot' or interleaved == 'cx':
            for i in range(matrix_size):
                a = m[i, :]
                for j in range(len(a) - 1):
                    qc.cx(a[j], a[j + 1])
        elif interleaved == 'cz':
            for i in range(matrix_size):
                a = m[i, :]
                for j in range(len(a) - 1):
                    qc.cz(a[j], a[j + 1])
        elif interleaved == 'rxx':
            for i in range(matrix_size):
                a = m[i, :]
                for j in range(len(a) - 1):
                    qc.rxx(matrix[i, i], a[j], a[j + 1])
        elif interleaved == 'ryy':
            for i in range(matrix_size):
                a = m[i, :]
                for j in range(len(a) - 1):
                    qc.ryy(matrix[i, i], a[j], a[j + 1])
        elif interleaved == 'rzz':
            for i in range(matrix_size):
                a = m[i, :]
                for j in range(len(a) - 1):
                    qc.rzz(matrix[i, i], a[j], a[j + 1])
        for i in range(matrix_size):
            for j in range(matrix_size):
                if (i < j) and (matrix[i, j] != 0.0):
                    if n_atom_to_qubit == 1:
                        q_c = m[i]
                        q_t = m[j]
                        if entangling_layer == 'rxx':
                            qc.rxx(matrix[i, j], q_c, q_t)
                        elif entangling_layer == 'ryy':
                            qc.ryy(matrix[i, j], q_c, q_t)
                        else:
                            qc.rzz(matrix[i, j], q_c, q_t)
                    else:
                        q_c = m[i, -1]
                        q_t = m[j, 0]
                        if entangling_layer == 'rxx':
                            qc.rxx(matrix[i, j], q_c, q_t)
                        elif entangling_layer == 'ryy':
                            qc.ryy(matrix[i, j], q_c, q_t)
                        else:
                            qc.rzz(matrix[i, j], q_c, q_t)

    
    return qc

def quantum_overlap(num_qubits, smiles_list, add_hydrogens: bool = False, entangling_layer: str = 'rzz', initial_layer: str = 'rx', n_atom_to_qubit: int = 1, interleaved: str = None, n_layers: int = 1) -> np.array:
    n = len(smiles_list)
    overlap_matrix = np.ones((n, n))
    s = np.zeros([len(smiles_list), 2**num_qubits],dtype=complex)
    for i in range(len(smiles_list)):
        coulomb_matrix_without_h = coulomb_matrix(smiles_list[i], add_hydrogens=add_hydrogens)
        circuit = matrix_to_circuit(coulomb_matrix_without_h, num_qubits, initial_layer=initial_layer, n_layers=n_layers, reverse_bits=True, entangling_layer=entangling_layer, n_atom_to_qubit=n_atom_to_qubit, interleaved=interleaved)
        statevector = np.array(Statevector(circuit))
        s[i] = statevector
    for i in range(len(smiles_list)):
        for j in range(len(smiles_list)):
            if i < j:
                overlap = np.abs(np.vdot(s[i], s[j]))**2
                overlap_matrix[i][j] = overlap
            elif i > j:
                overlap_matrix[i][j] = overlap_matrix[j][i]
    return overlap_matrix



def quantum_sampler_overlap(num_shots, backend, smiles_list, add_hydrogens: bool = False, entangling_layer: str = 'rzz', initial_layer: str = 'rx', n_atom_to_qubit: int = 1, interleaved: str = None, n_layers: int = 1) -> np.array:
    n = len(smiles_list)
    overlap_matrix = np.zeros((n, n))
    for i in range(len(smiles_list)):
        for j in range(len(smiles_list)):
            if i <= j:
                matrix_A = coulomb_matrix(smiles_list[i], add_hydrogens=add_hydrogens)
                matrix_B = coulomb_matrix(smiles_list[j], add_hydrogens=add_hydrogens)
                max_size = max(matrix_A.shape[0], matrix_B.shape[0])
                total_qubits = max_size * n_atom_to_qubit
                circuit_A = matrix_to_circuit(matrix_A, total_qubits, initial_layer=initial_layer, n_layers=n_layers, reverse_bits=True, entangling_layer=entangling_layer, n_atom_to_qubit=n_atom_to_qubit, interleaved=interleaved)
                circuit_B = matrix_to_circuit(matrix_B, total_qubits, initial_layer=initial_layer, n_layers=n_layers, reverse_bits=True, entangling_layer=entangling_layer, n_atom_to_qubit=n_atom_to_qubit, interleaved=interleaved)
                compiled_circuit = UnitaryOverlap(circuit_A, circuit_B)
                compiled_circuit.measure_all()
                pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
                transpile_circuit = pm.run(compiled_circuit)
                sampler = Sampler(mode=backend, options={"default_shots": int(num_shots)})
                job = sampler.run([transpile_circuit])
                result = job.result()
                pub_result = result[0]
                counts = pub_result.data.meas.get_counts()
                if '0'*total_qubits not in counts:
                    prob_0 = 0
                else:
                    prob_0 = counts['0'*total_qubits] / num_shots
                overlap_matrix[i][j] = prob_0
            else:
                overlap_matrix[i][j] = overlap_matrix[j][i]
    return overlap_matrix

def save_to_excel(df, file_path, sheet_name):
    try:
        # If the file exists, append the DataFrame to the existing file
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    except FileNotFoundError:
        # If the file does not exist, create a new file and save the DataFrame
        df.to_excel(file_path, sheet_name=sheet_name, index=False, header=False)

def remove_idle_qubits(circuit):
    dag = circuit_to_dag(circuit)
    idle_wires = list(dag.idle_wires())

    for wire in idle_wires:
        if wire in dag.qubits:
            dag.remove_qubits(wire)

    # Do NOT assign to dag.qregs — Qiskit handles this internally
    return dag_to_circuit(dag)

def generate_overlap_circuit(smiles0, smiles1, num0, num1, feature_map_config):
    num_qubits = max(num0, num1)

    # Encode the molecules into quantum circuits using BondFeatureMap
    circuit_A = BondFeatureMap(BondOrderMatrix().compute(smiles0), num_qubits=num_qubits, **feature_map_config)
    
    circuit_B = BondFeatureMap(BondOrderMatrix().compute(smiles1), num_qubits=num_qubits, **feature_map_config)



    # Create a quantum circuit to compute overlap between two molecular encodings
    compiled_circuit = UnitaryOverlap(circuit_A, circuit_B, barrier=False, measure_all=False).decompose()
    


    service = QiskitRuntimeService()

    # Quantum Device
    device = "ibm_fez"
    backend = service.backend(device)

    

    pass_manager = generate_preset_pass_manager(3, backend=backend)
    pass_manager_aer = generate_preset_pass_manager(3, backend=AerSimulator())
    transpiled_circuit = pass_manager.run(compiled_circuit)
    
    transpiled_circuit = remove_idle_qubits(transpiled_circuit)

    transpiled_circuit.measure_all()
    
    transpiled_circuit = pass_manager_aer.run(transpiled_circuit)

    
    
    # transpiled_circuit = pass_manager.run(compiled_circuit)
    # display(transpiled_circuit.draw(output='mpl', idle_wires=False))
    return transpiled_circuit

# Function to save molecule as high resolution PDF using cairosvg
def molecule_to_pdf(mol, file_name, directory, width=1200, height=1200):
    """Save substance structure as PDF"""

    # Define full path name
    full_path = os.path.join(directory, f"{file_name}.pdf")

    # Render high resolution molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Export to pdf
    cairosvg.svg2pdf(bytestring=drawer.GetDrawingText().encode(), write_to=full_path)