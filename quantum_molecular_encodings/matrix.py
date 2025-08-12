import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from PIL.Image import Image
from quantum_molecular_encodings.processing import find_rs_stereoisomers, find_ze_conformers

AVERAGE_BOND_LENGTHS: dict[tuple[int, int, Chem.BondType], float] = {
    (1, 6, Chem.BondType.SINGLE): 1.09,  # H-C single bond
    (1, 7, Chem.BondType.SINGLE): 1.01,  # H-N single bond
    (1, 8, Chem.BondType.SINGLE): 0.96,  # H-O single bond
    (1, 16, Chem.BondType.SINGLE): 1.34,  # H-S single bond
    (6, 6, Chem.BondType.SINGLE): 1.54,  # C-C single bond
    (6, 6, Chem.BondType.DOUBLE): 1.34,  # C=C double bond
    (6, 6, Chem.BondType.TRIPLE): 1.20,  # C≡C triple bond
    (6, 6, Chem.BondType.AROMATIC): 1.41,  # C-C aromatic bond
    (6, 8, Chem.BondType.SINGLE): 1.43,  # C-O single bond
    (6, 8, Chem.BondType.DOUBLE): 1.23,  # C=O double bond
    (6, 8, Chem.BondType.TRIPLE): 1.13,  # C≡O triple bond
    (6, 8, Chem.BondType.AROMATIC): 1.38,  # C-O aromatic bond
    (6, 7, Chem.BondType.SINGLE): 1.47,  # C-N single bond
    (6, 7, Chem.BondType.DOUBLE): 1.25,  # C=N double bond
    (6, 7, Chem.BondType.TRIPLE): 1.15,  # C≡N triple bond
    (6, 7, Chem.BondType.AROMATIC): 1.35,  # C-N aromatic bond
    (6, 7, Chem.BondType.AROMATIC): 1.35,  # C-N aromatic bond
    (6, 16, Chem.BondType.SINGLE): 1.81,  # C-S single bond
    (6, 16, Chem.BondType.DOUBLE): 1.60,  # C=S double bond
    (6, 16, Chem.BondType.AROMATIC): 1.70,  # C-S aromatic bond
    (8, 8, Chem.BondType.SINGLE): 1.48,  # O-O single bond
    (8, 8, Chem.BondType.DOUBLE): 1.21,  # O=O double bond
    (8, 8, Chem.BondType.DOUBLE): 1.35,  # O-O aromatic bond
    (7, 7, Chem.BondType.SINGLE): 1.47,  # N-N single bond
    (7, 7, Chem.BondType.DOUBLE): 1.24,  # N=N double bond
    (7, 7, Chem.BondType.TRIPLE): 1.10,  # N≡N triple bond
    (7, 7, Chem.BondType.AROMATIC): 1.36,  # N-N aromatic bond
    (7, 8, Chem.BondType.SINGLE): 1.44,  # N-O single bond
    (7, 8, Chem.BondType.DOUBLE): 1.20,  # N=O double bond
    (7, 8, Chem.BondType.AROMATIC): 1.32,  # N-O aromatic bond
    (16, 16, Chem.BondType.SINGLE): 2.05,  # S-S single bond
    (16, 8, Chem.BondType.SINGLE): 1.58,  # S-O single bond
    (16, 8, Chem.BondType.DOUBLE): 1.42,  # S=O double bond
    (16, 8, Chem.BondType.AROMATIC): 1.50,  # S-O aromatic bond
    # Add more bonds as needed
}

BOND_ORDER: dict[Chem.BondType, float] = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 1.5,
    # Add more bonds as needed
}


class BaseMatrix:
    def __init__(self, bond_coupling: float = 1.0):
        """
        Base class for computing molecular matrices.

        Parameters:
        - bond_coupling (float): A factor to scale the bond interaction values.
        """
        self.bond_coupling = bond_coupling
        self.molecule = None

    def compute(self, smiles: str, add_hydrogens: bool = False, exponent: float = 3.0) -> np.ndarray:
        """
        Computes the molecular matrix for a given molecule specified by a SMILES string.

        Parameters:
        - smiles (str): The SMILES string representing the molecule.
        - add_hydrogens (bool): Whether to add hydrogen atoms to the molecule.

        Returns:
        - np.ndarray: The molecular matrix of the molecule.
        """
        molecule = Chem.MolFromSmiles(smiles)
        if add_hydrogens:
            molecule = Chem.AddHs(molecule)

        self.molecule = molecule  # Save the RDKit molecule object

        atomic_numbers = np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
        num_atoms = len(atomic_numbers)
        matrix = np.zeros((num_atoms, num_atoms))

        # Precompute diagonal elements
        diagonal_elements = 0.5 * atomic_numbers ** exponent

        # Find stereoisomers
        isomers = find_rs_stereoisomers(smiles)
        s_stereoisomers = isomers['S'] # if 'S' isomer, multiply by -1.
        
        # r_stereoisomers = isomers['R'] # if 'R' isomer, multiply by 1.

        # u_stereoisomers = isomers['U'] # unasigned stereoisomers
        
        for idx in s_stereoisomers:
            #print(f"S isomer at {idx}")
            diagonal_elements[idx] *= -1
        
        np.fill_diagonal(matrix, diagonal_elements)

        # Fill the matrix based on bond distances or bond orders
        matrix = self._fill_matrix(molecule, smiles, atomic_numbers, matrix)

        return matrix

    def _fill_matrix(self, molecule: Chem.Mol, smiles: str, atomic_numbers: np.ndarray,
                     matrix: np.ndarray) -> np.ndarray:
        """
        Abstract method to fill the matrix based on specific criteria.
        Must be implemented by subclasses.

        Parameters:
        - molecule (Chem.Mol): The RDKit molecule object.
        - atomic_numbers (np.ndarray): Array of atomic numbers.
        - matrix (np.ndarray): The matrix to be filled.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def draw_molecule(self, file_path: str = None) -> Image:
        """
        Draws the image of the molecule and optionally saves it to a file.

        Parameters:
        - file_path (str, optional): Path to save the image file. If None, the image is displayed.

        Returns:
        - PIL.Image.Image: Image of the molecule.
        """
        if self.molecule is None:
            raise ValueError("Molecule not initialized. Run `compute` method first.")

        # Draw the molecule image
        image = Draw.MolToImage(self.molecule)

        if file_path:
            image.save(file_path)

        return image


class CoulombMatrix(BaseMatrix):
    def __init__(self, bond_coupling: float = 1.0):
        """
        Class for computing Coulomb matrices using average bond lengths.

        Parameters:
        - bond_coupling (float): A factor to scale the bond interaction values.

        Example Usage:

        .. code-block:: python

            # Initialize CoulombMatrix instance
            cm = CoulombMatrix()

            # Compute Coulomb matrix for a water molecule (H2O)
            smiles = "O"
            matrix = cm.compute(smiles, add_hydrogens=True)

            print(matrix)

        """
        super().__init__(bond_coupling)
        self.average_bond_lengths: dict[tuple[int, int, Chem.BondType], float] = (
            AVERAGE_BOND_LENGTHS)

    def _fill_matrix(self, molecule: Chem.Mol, smiles: str, atomic_numbers: np.ndarray,
                     matrix: np.ndarray) -> np.ndarray:
        """
        Fills the matrix using average bond lengths.

        Parameters:
        - molecule (Chem.Mol): The RDKit molecule object.
        - smiles (str): The SMILES string of the molecule.
        - atomic_numbers (np.ndarray): Array of atomic numbers.
        - matrix (np.ndarray): The matrix to be filled.
        """
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            atom_pair = (
                min(atomic_numbers[i], atomic_numbers[j]),
                max(atomic_numbers[i], atomic_numbers[j]),
                bond_type
            )
            distance = self.average_bond_lengths.get(atom_pair)

            if distance is None:
                raise ValueError(f"Bond length not defined for atom pair: {atom_pair}")

            matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j] / distance) * self.bond_coupling
            matrix[j, i] = matrix[i, j]

        return matrix


class BondOrderMatrix(BaseMatrix):
    def __init__(self, bond_coupling: float = 1.0):
        """
        Class for computing matrices using bond orders.

        Parameters:
        - bond_coupling (float): A factor to scale the bond interaction values.

        Example Usage:

        .. code-block:: python

            # Initialize BondOrderMatrix instance
            bom = BondOrderMatrix()

            # Compute BondOrder matrix for benzene (C6H6)
            smiles = "c1ccccc1"
            matrix = bom.compute(smiles, add_hydrogens=True)

            print(matrix)

        """
        super().__init__(bond_coupling)
        self.bond_orders: dict[Chem.BondType, float] = BOND_ORDER

    def _fill_matrix(self, molecule: Chem.Mol, smiles: str, atomic_numbers: np.ndarray,
                     matrix: np.ndarray) -> np.ndarray:
        """
        Fills the matrix using bond orders.

        Parameters:
        - molecule (Chem.Mol): The RDKit molecule object.
        - smiles (str): The SMILES string of the molecule.
        - atomic_numbers (np.ndarray): Array of atomic numbers.
        - matrix (np.ndarray): The matrix to be filled.
        """

        z_conformers = find_ze_conformers(smiles)['Z']
        
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_order = self.bond_orders.get(bond_type)

            if bond_order == 2 and (i,j) in z_conformers:
                bond_order = -1 * bond_order

            if bond_order is None:
                raise ValueError(f"Bond order not defined for bond type: {bond_type}")

            matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j] / bond_order) * self.bond_coupling
            matrix[j, i] = matrix[i, j]

        return matrix
