import pandas as pd
from rdkit import Chem
import itertools

def get_unique_smiles(smiles):
    # Convert SMILES string to a molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    # Generate the unique SMILES using CANGEN algorithm
    unique_smiles = Chem.MolToSmiles(mol, canonical=True)
    
    return unique_smiles

def reverse_smiles(smiles):
    # Generate RDKit molecule from SMILES string
    mol = Chem.MolFromSmiles(smiles)
    
    # Reverse the atom order
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    
    # Generate the reversed SMILES string
    reversed_smiles = Chem.MolToSmiles(reversed_mol, canonical=False)

    canonical_input = Chem.CanonSmiles(smiles)
    canonical_output = Chem.CanonSmiles(reversed_smiles)

    if canonical_input != canonical_output:
        print("Conversion failed")
        return
    
    return reversed_smiles

def find_rs_stereoisomers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    chiral_cc = Chem.FindMolChiralCenters(mol, includeUnassigned=True)  

    # if not len(chiral_cc) == 0:
    #     print(chiral_cc)
    rs_stereoisomers = {'R': [], 'S': [], 'U': []}  
    
    for idx, _ in chiral_cc:
        atom = mol.GetAtomWithIdx(idx)
        chiral_tag = atom.GetChiralTag()
        if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            rs_stereoisomers['R'].append(idx)
        elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            rs_stereoisomers['S'].append(idx)
        elif chiral_tag == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            rs_stereoisomers['U'].append(idx)
    
    return rs_stereoisomers

def find_ze_conformers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ze_conformers = {'Z': [], 'E': []}
    
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            stereo = bond.GetStereo()
            
            if stereo == Chem.BondStereo.STEREOZ:
                ze_conformers['Z'].append((idx1, idx2))
            elif stereo == Chem.BondStereo.STEREOE:
                ze_conformers['E'].append((idx1, idx2))
    
    return ze_conformers

def add_rs_stereochemistry(smiles):
    bootstrapped_smiles_array = []
    rs_stereoisomers = find_rs_stereoisomers(smiles)
    unspecified_indices = rs_stereoisomers['U']

    # Generate all combinations of '@' and '@@' for unspecified indices
    combinations = list(itertools.product(['@', '@@'], repeat=len(unspecified_indices)))
    
    for combo in combinations:
        mol = Chem.MolFromSmiles(smiles)
        for idx, chirality in zip(unspecified_indices, combo):
            atom = mol.GetAtomWithIdx(idx)
            if chirality == '@':
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif chirality == '@@':
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
    
        new_smiles = Chem.MolToSmiles(mol)
        new_smiles = reverse_smiles(new_smiles)
        bootstrapped_smiles_array.append(new_smiles)

    return bootstrapped_smiles_array

def bootstrap_smiles_list(smiles_input):
    new_smiles_array = []

    # Ensure smiles_input is a list
    if isinstance(smiles_input, str):
        smiles_list = [smiles_input]
    else:
        smiles_list = smiles_input

    for smiles in smiles_list:
        rs_stereoisomers = find_rs_stereoisomers(smiles)
        unspecified_indices = rs_stereoisomers['U']

        if unspecified_indices:
            print(f"Unspecified stereochemistry found in {smiles}")
            # Generate new SMILES strings with assigned stereochemistries
            new_smiles_with_stereochemistry = add_rs_stereochemistry(smiles)
            
            new_smiles_array.extend(new_smiles_with_stereochemistry)
        else:
            # If no unassigned stereochemistries, add the original SMILES
            new_smiles_array.append(smiles)

    

    return new_smiles_array

def count_atoms_and_check_validity(smiles: str, has_triple_bond: bool, has_isotopes: bool, num_double_bonds: int):
    """
    Count the number of carbon and oxygen atoms in a SMILES string and check for valid atoms (C, H, O).
    Additionally, check for the presence of triple bonds, isotopes, and a specified number of double bonds.

    Parameters:
    smiles (str): The SMILES string representing the molecule.
    has_triple_bond (bool): Whether the molecule should have a triple bond.
    has_isotopes (bool): Whether the molecule should have isotopes (D or 13C).
    num_double_bonds (int): The specified number of double bonds the molecule should have.

    Returns:
    tuple: A tuple containing the following:
        - carbon_count (int): The number of carbon atoms in the molecule.
        - oxygen_count (int): The number of oxygen atoms in the molecule.
        - valid_atoms (bool): Whether the molecule contains only valid atoms (C, H, O).
        - has_triple_bond (bool): Whether the molecule has a triple bond.
        - has_isotopes (bool): Whether the molecule has isotopes (D or 13C).
        - correct_double_bonds (bool): Whether the molecule has the specified number of double bonds.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0, 0, False, False, False, False
    
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    oxygen_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
    valid_atoms = all(atom.GetSymbol() in ['C', 'H', 'O'] for atom in mol.GetAtoms())
    actual_triple_bond = any(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE for bond in mol.GetBonds())
    actual_isotopes = any(atom.GetIsotope() in [2, 13] for atom in mol.GetAtoms())
    actual_double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)
    
    correct_double_bonds = (actual_double_bonds == num_double_bonds)
    
    return carbon_count, oxygen_count, valid_atoms, actual_triple_bond, actual_isotopes, correct_double_bonds

# Function to check if a molecule contains a COOH group (carboxylic acid)
def contains_cooh_group(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    # Updated SMARTS pattern for carboxylic acid that does not rely on explicit hydrogen atoms
    cooh_smarts = Chem.MolFromSmarts('C(=O)O')
    matches = mol.GetSubstructMatches(cooh_smarts)
    
    # Check if the COOH group is not part of an ester
    for match in matches:
        carbon, oxygen1, oxygen2 = match
        if mol.GetAtomWithIdx(oxygen2).GetSymbol() == 'O' and mol.GetAtomWithIdx(oxygen2).GetDegree() == 1:
            return True
    return False