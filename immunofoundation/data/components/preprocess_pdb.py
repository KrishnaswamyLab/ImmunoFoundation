from Bio.PDB import PDBParser, MMCIFParser
from Bio.SeqUtils import seq1
import gzip
import numpy as np
import io


def _read_file_text(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return f.read()
    else:
        with open(path, 'r') as f:
            return f.read()


def extract_ca_and_sequence(pdb_file):
    """Robustly extract CA coordinates and sequence from mmCIF or PDB files.

    Tries MMCIFParser first (for mmCIF), and falls back to PDBParser when that
    fails or the file appears to be a PDB file. Accepts gzipped files as well.
    Returns: (ca_coords_peptide, sequence_peptide, ca_coords_mhc, sequence_mhc)
    """
    mmcif_parser = MMCIFParser(QUIET=True)
    pdb_parser = PDBParser(QUIET=True)

    text = None
    try:
        text = _read_file_text(pdb_file)
    except Exception as e:
        raise ValueError(f"Could not read structure file {pdb_file}: {e}")

    # Heuristic: mmCIF files usually start with 'data_' directive
    is_mmcif = text.lstrip().startswith('data_')

    structure = None
    if is_mmcif:
        try:
            # MMCIFParser can accept a file handle
            fh = io.StringIO(text)
            structure = mmcif_parser.get_structure('protein', fh)
        except Exception:
            structure = None

    if structure is None:
        # try PDB parser
        try:
            fh = io.StringIO(text)
            structure = pdb_parser.get_structure('protein', fh)
        except Exception as e:
            # give a helpful error
            raise ValueError(f"Failed to parse structure file {pdb_file} as mmCIF or PDB: {e}")

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
                except Exception:
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
                except Exception:
                    sequence_mhc.append('X')

    sequence_peptide = ''.join(sequence_peptide)
    sequence_mhc = ''.join(sequence_mhc)

    return ca_coords_peptide, sequence_peptide, ca_coords_mhc, sequence_mhc


def normalize_coords(coords):
    """Normalize coordinates to [-1, 1]."""
    coords = np.array(coords)
    if coords.size == 0:
        return coords
    minv = coords.min()
    maxv = coords.max()
    if maxv == minv:
        return coords - minv
    return 2 * (coords - minv) / (maxv - minv) - 1
