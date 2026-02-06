from Bio.PDB import PDBParser, MMCIFParser
from Bio.SeqUtils import seq1
import gzip
import numpy as np

parser = MMCIFParser(QUIET=True)
    

def extract_ca_and_sequence(pdb_file):
    parser = MMCIFParser(QUIET=True)
    if pdb_file.endswith(".gz"):
        with gzip.open(pdb_file, 'rt') as f:
            structure = parser.get_structure('protein', f)
    else:
        structure = parser.get_structure('protein', pdb_file)
    model = structure[0]
    
    ca_coords_peptide = []
    sequence_peptide = []
    if 'A' in model:
        for residue in model['A']:
            if 'CA' in residue:
                ca_atom = residue['CA']
                ca_coords_peptide.append(tuple(ca_atom.get_coord()))
                
                try:
                    aa = seq1(residue.get_resname())
                    sequence_peptide.append(aa)
                except KeyError:
                    # Handle non-standard residues
                    sequence_peptide.append('X')
    
    ca_coords_mhc = []
    sequence_mhc = []
    
    if 'B' in model:
        for residue in model['B']:
            if 'CA' in residue:
                ca_atom = residue['CA']
                ca_coords_mhc.append(tuple(ca_atom.get_coord()))
                
                try:
                    aa = seq1(residue.get_resname())
                    sequence_mhc.append(aa)
                except KeyError:
                    # Handle non-standard residues
                    sequence_mhc.append('X')
    
    sequence_peptide = ''.join(sequence_peptide)
    sequence_mhc = ''.join(sequence_mhc)

    # Normalize ca_coords to [-1, 1] separately
    if len(ca_coords_peptide):
        pep = np.array(ca_coords_peptide)
        ca_coords_peptide = (2 * (pep - pep.min()) / (pep.max() - pep.min()) - 1).tolist()
    if len(ca_coords_mhc):
        mhc = np.array(ca_coords_mhc)
        ca_coords_mhc = (2 * (mhc - mhc.min()) / (mhc.max() - mhc.min()) - 1).tolist()

    return ca_coords_peptide, sequence_peptide, ca_coords_mhc, sequence_mhc
