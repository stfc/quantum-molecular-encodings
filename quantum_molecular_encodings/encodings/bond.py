from qiskit import QuantumCircuit
import numpy as np

INTERLEAVED_ALLOWED_GATES: dict[str, list[str]] = {
    'controlled': ['cnot', 'cx', 'cz'],
    'rotations': ['rxx', 'ryy', 'rzz']
}

class BondFeatureMap(QuantumCircuit):
    """
    Converts a matrix to a QuantumCircuit object.

    Parameters:
    - matrix (np.ndarray): The matrix to convert.
    - num_qubits (int): Number of qubits in the circuit.
    - n_layers (int): Number of layers in the circuit.
    - reverse_bits (bool): Whether to reverse the bits in the circuit.
    - initial_layer (str): Type of initial layer gates ('rx', 'ry', 'rz').
    - entangling_layer (str): Type of entangling gates ('rxx', 'ryy', 'rzz').
    - n_atom_to_qubit (int): Number of qubits per matrix element.
    - interleaved (str): Type of interleaving gates ('None', 'cnot', 'cz').
    """

    def __init__(self,
                 matrix: np.ndarray,
                 num_qubits: int,
                 n_layers: int = 1,
                 reverse_bits: bool = False,
                 initial_layer: str = 'rx',
                 entangling_layer: str = 'rzz',
                 n_atom_to_qubit: int = 1,
                 interleaved: str | None = None) -> None:
        super().__init__(num_qubits)

        indices = self._initialize_qubit_indices(
            matrix.shape[0], num_qubits, n_atom_to_qubit, reverse_bits
        )

        for _ in range(n_layers):
            self._apply_initial_layer(matrix, indices, initial_layer, n_atom_to_qubit)

            if interleaved is None:
                pass
            elif interleaved.lower() in INTERLEAVED_ALLOWED_GATES['rotations']:
                self._apply_interleaving_rotation(matrix, indices, interleaved)
            elif interleaved.lower() in INTERLEAVED_ALLOWED_GATES['controlled']:
                self._apply_interleaving_controlled(indices, interleaved)
            else:
                raise ValueError(f"Only the following interleaving gates are allowed: "
                                 f"{INTERLEAVED_ALLOWED_GATES.values()}. Got {interleaved} "
                                 f"instead.")

            self._apply_entangling_layers(matrix, indices, entangling_layer, n_atom_to_qubit)

    @staticmethod
    def _initialize_qubit_indices(matrix_size: int, num_qubits: int, n_atom_to_qubit: int,
                                  reverse_bits: bool) -> np.ndarray:
        """
        Initializes qubit indices for the quantum circuit based on matrix size and bit-reversal
        option.

        Parameters:
        - matrix_size (int): Size of the matrix (number of qubits in the matrix).
        - num_qubits (int): Total number of qubits in the circuit.
        - n_atom_to_qubit (int): Number of qubits per matrix element.
        - reverse_bits (bool): Whether to reverse the qubit indices.

        Returns:
        - np.ndarray: Array of qubit indices.
        """
        if reverse_bits:
            indices = np.flip(np.arange(num_qubits - matrix_size * n_atom_to_qubit, num_qubits))
        else:
            indices = np.arange(0, matrix_size * n_atom_to_qubit)

        return np.reshape(indices, (matrix_size, n_atom_to_qubit))

    def _apply_initial_layer(self, matrix: np.ndarray, indices: np.ndarray,
                             initial_layer: str, n_atom_to_qubit: int) -> None:
        """
        Applies initial rotation gates to the quantum circuit.

        Parameters:
        - qc (QuantumCircuit): QuantumCircuit object to which gates will be applied.
        - matrix (np.ndarray): Matrix providing the rotation angles.
        - indices (np.ndarray): Qubit indices for applying gates.
        - initial_layer (str): Type of initial rotation gates ('rx', 'ry', 'rz').
        - n_atom_to_qubit (int): Number of qubits per matrix element.
        """
        rotation_gates = {
            'ry': self.ry,
            'rz': self.rz,
            'rx': self.rx
        }

        apply_rotation = rotation_gates.get(initial_layer, None)
        if apply_rotation is None:
            raise ValueError(f"Unsupported initial layer gate: {initial_layer}")

        for i in range(matrix.shape[0]):
            for k in range(n_atom_to_qubit):
                apply_rotation(matrix[i, i], indices[i, k])

    def _apply_interleaving_controlled(self,
                                       indices: np.ndarray,
                                       interleaved: str) -> None:
        """
        Applies interleaving controlled gates (CNOT or CZ) to the quantum circuit based on the
        specified gate type.

        Parameters:
        - indices (np.ndarray): A 2D array of qubit indices for applying the gates.
                                Each row contains the indices of qubits for a specific
                                interleaving operation.
        - interleaved (str): Type of interleaving gate to apply. Supported values are:
                             'cnot' or 'cx' for controlled-NOT (CNOT) gates,
                             'cz' for controlled-Z (CZ) gates.
        """
        gate_func = self.cx if interleaved in ('cnot', 'cx') else self.cz

        for qubits in indices:
            for q1, q2 in zip(qubits[:-1], qubits[1:]):
                gate_func(q1, q2)

    def _apply_interleaving_rotation(self,
                                     matrix: np.ndarray,
                                     indices: np.ndarray,
                                     interleaved: str) -> None:
        """
        Applies interleaving rotation gates (RXX, RYY, RZZ) to the quantum circuit based on the
        specified gate type.

        Parameters:
        - matrix (np.ndarray): A matrix providing the entangling angles for the interleaving
                                rotation gates.
                               The diagonal elements of the matrix correspond to the angles used.
        - indices (np.ndarray): A 2D array of qubit indices for applying the gates.
                                Each row contains the indices of qubits for a specific
                                interleaving operation.
        - interleaved (str): Type of interleaving rotation gate to apply. Supported values are:
                             'rxx' for RXX gates,
                             'ryy' for RYY gates,
                             'rzz' for RZZ gates.
        """
        gate_func = {
            'rxx': self.rxx,
            'ryy': self.ryy,
            'rzz': self.rzz
        }.get(interleaved)

        for i, qubits in enumerate(indices):
            angle = matrix[i, i]
            for q1, q2 in zip(qubits[:-1], qubits[1:]):
                gate_func(angle, q1, q2)

    def _apply_entangling_layers(self, matrix: np.ndarray, indices: np.ndarray,
                                 entangling_layer: str, n_atom_to_qubit: int) -> None:
        """
        Applies entangling gates to the quantum circuit.

        Parameters:
        - qc (QuantumCircuit): QuantumCircuit object to which gates will be applied.
        - matrix (np.ndarray): Matrix providing the entangling angles.
        - indices (np.ndarray): Qubit indices for applying gates.
        - entangling_layer (str): Type of entangling gates ('rxx', 'ryy', 'rzz').
        - n_atom_to_qubit (int): Number of qubits per matrix element.
        """
        entangling_gates = {
            'rxx': self.rxx,
            'ryy': self.ryy,
            'rzz': self.rzz
        }

        apply_entangle = entangling_gates.get(entangling_layer, None)
        if apply_entangle is None:
            raise ValueError(f"Unsupported entangling layer gate: {entangling_layer}")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i < j and matrix[i, j] != 0.0 and i % 2 == 0:
                    q_c = indices[i, -1] if n_atom_to_qubit > 1 else indices[i]
                    q_t = indices[j, 0] if n_atom_to_qubit > 1 else indices[j]
                    apply_entangle(matrix[i, j], q_c, q_t)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i < j and matrix[i, j] != 0.0 and i % 2 == 1:
                    q_c = indices[i, -1] if n_atom_to_qubit > 1 else indices[i]
                    q_t = indices[j, 0] if n_atom_to_qubit > 1 else indices[j]
                    apply_entangle(matrix[i, j], q_c, q_t)
