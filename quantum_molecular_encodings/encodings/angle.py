from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import CXGate, CZGate
from qiskit.circuit.library import NLocal
from qiskit.circuit import ParameterVector


class AngleEncodingCircuit(NLocal):
    """Angle Encoding Quantum Circuit.

    This class creates a quantum feature map where classical data is encoded into quantum states
    using rotation gates and entangling gates. The circuit structure can be repeated multiple times
    (reps), and the entanglement pattern between qubits can be customized.

    Args:
        feature_dimension (Optional[int]): Number of qubits (and features) in the circuit.
        reps (int): The number of repeated blocks of the circuit.
        rotation_gate (str): The type of rotation gate to use ('rx', 'ry', 'rz').
        rotation_parameters (Optional[List[float]]): A list of angles to use for the rotation gates.
        entangling_gate (str): The entangling gate to use ('cx', 'cz').
        entanglement (str): The entanglement pattern between qubits ('linear', 'reverse_linear',
            'sca', 'circular', 'full').
        parameter_prefix (str): The prefix to use for the rotation parameters.
        insert_barriers (bool): Whether to insert barriers between circuit layers.
        name (str): The name of the quantum circuit.

    Example:

        circuit = AngleEncodingCircuit(4,
                                       reps=2,
                                       rotation_parameters=None,
                                       entanglement="circular",
                                       insert_barriers=True)
        print(circuit.decompose())

             ┌──────────┐ ░ ┌───┐                ░ ┌──────────┐ ░ ┌───┐
        q_0: ┤ Ry(x[0]) ├─░─┤ X ├──■─────────────░─┤ Ry(x[4]) ├─░─┤ X ├──■────────────
             ├──────────┤ ░ └─┬─┘┌─┴─┐           ░ ├──────────┤ ░ └─┬─┘┌─┴─┐
        q_1: ┤ Ry(x[1]) ├─░───┼──┤ X ├──■────────░─┤ Ry(x[5]) ├─░───┼──┤ X ├──■───────
             ├──────────┤ ░   │  └───┘┌─┴─┐      ░ ├──────────┤ ░   │  └───┘┌─┴─┐
        q_2: ┤ Ry(x[2]) ├─░───┼───────┤ X ├──■───░─┤ Ry(x[6]) ├─░───┼───────┤ X ├──■──
             ├──────────┤ ░   │       └───┘┌─┴─┐ ░ ├──────────┤ ░   │       └───┘┌─┴─┐
        q_3: ┤ Ry(x[3]) ├─░───■────────────┤ X ├─░─┤ Ry(x[7]) ├─░───■────────────┤ X ├
             └──────────┘ ░                └───┘ ░ └──────────┘ ░                └───┘

        # Now assign parameters in-place
        circuit.assign_parameters([0, 1, 2, 3, 4, 5, 6, 7], inplace=True)
        print(circuit.decompose())

             ┌───────┐ ░ ┌───┐                ░ ┌───────┐ ░ ┌───┐
        q_0: ┤ Ry(0) ├─░─┤ X ├──■─────────────░─┤ Ry(4) ├─░─┤ X ├──■────────────
             ├───────┤ ░ └─┬─┘┌─┴─┐           ░ ├───────┤ ░ └─┬─┘┌─┴─┐
        q_1: ┤ Ry(1) ├─░───┼──┤ X ├──■────────░─┤ Ry(5) ├─░───┼──┤ X ├──■───────
             ├───────┤ ░   │  └───┘┌─┴─┐      ░ ├───────┤ ░   │  └───┘┌─┴─┐
        q_2: ┤ Ry(2) ├─░───┼───────┤ X ├──■───░─┤ Ry(6) ├─░───┼───────┤ X ├──■──
             ├───────┤ ░   │       └───┘┌─┴─┐ ░ ├───────┤ ░   │       └───┘┌─┴─┐
        q_3: ┤ Ry(3) ├─░───■────────────┤ X ├─░─┤ Ry(7) ├─░───■────────────┤ X ├
             └───────┘ ░                └───┘ ░ └───────┘ ░                └───┘
    """

    def __init__(
        self,
        feature_dimension: int | None = None,
        reps: int = 1,
        rotation_gate: str = 'ry',
        rotation_parameters: list[float] | ndarray | None = None,
        entangling_gate: str = 'cx',
        entanglement: str = "linear",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        skip_final_rotation_layer: bool = True,
        name: str = "AngleEncodingFeatureMap",
    ) -> None:
        """Initialize the angle encoding circuit.

        Args:
            feature_dimension: Number of qubits/features in the circuit.
            reps: Number of repetitions of the circuit block.
            rotation_gate: Rotation gate to use ('rx', 'ry', 'rz').
            rotation_parameters: List of angles for the rotation gates.
            entangling_gate: Entangling gate to use ('cx', 'cz').
            entanglement: Entanglement pattern for qubits ('linear', 'circular', 'full').
            parameter_prefix: Prefix for rotation parameters.
            insert_barriers: Whether to insert barriers between layers.
            name: Name of the quantum circuit.
        """
        if rotation_parameters is None:
            rotation_parameters = ParameterVector("_", length=feature_dimension)

        # Create the rotation block with the selected rotation gate
        rotation_block = QuantumCircuit(feature_dimension)
        r_gate = {
            'rx': rotation_block.rx,
            'ry': rotation_block.ry,
            'rz': rotation_block.rz
        }.get(rotation_gate)
        if r_gate is None:
            raise ValueError(f"Unsupported rotation gate: {rotation_gate}")

        for i, angle in enumerate(rotation_parameters):
            r_gate(angle, i)

        # Create the entanglement block with the selected entangling gate
        entanglement_block = {
            'cx': CXGate(),
            'cz': CZGate()
        }.get(entangling_gate)
        if entanglement_block is None:
            raise ValueError(f"Unsupported entangling gate: {entangling_gate}")

        # Initialize the NLocal circuit
        super().__init__(
            num_qubits=feature_dimension,
            reps=reps,
            rotation_blocks=rotation_block,
            entanglement_blocks=entanglement_block,
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            skip_final_rotation_layer=skip_final_rotation_layer,
            name=name,
        )