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

    return ca_coords_peptide, sequence_peptide, ca_coords_mhc, sequence_mhc


def extract_ca_and_sequence_pmhc(pdb_file, peptide_chain='P', mhc_chain='A'):
    is_pdb = pdb_file.endswith(".pdb") or pdb_file.endswith(".pdb.gz")
    parser = PDBParser(QUIET=True) if is_pdb else MMCIFParser(QUIET=True)
    if pdb_file.endswith(".gz"):
        with gzip.open(pdb_file, 'rt') as f:
            structure = parser.get_structure('protein', f)
    else:
        structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    def _read_chain(chain_id):
        coords, seq = [], []
        if chain_id in model:
            for residue in model[chain_id]:
                if 'CA' in residue:
                    coords.append(tuple(residue['CA'].get_coord()))
                    try:
                        seq.append(seq1(residue.get_resname()))
                    except KeyError:
                        seq.append('X')
        return coords, ''.join(seq)

    pep_coords, pep_seq = _read_chain(peptide_chain)
    mhc_coords, mhc_seq = _read_chain(mhc_chain)
    return pep_coords, pep_seq, mhc_coords, mhc_seq


def normalize_coords(coords):
    """Normalize coordinates to [-1, 1]."""
    coords = np.array(coords)
    return 2 * (coords - coords.min()) / (coords.max() - coords.min()) - 1
