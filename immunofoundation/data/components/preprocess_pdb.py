from cpdb import PDB

def extract_protein_data(pdb_file):
    pdb = PDB(pdb_file)
    chain = pdb.chains[0]
    
    ca_coords = []
    sequence = ""
    
    for residue in chain.residues:
        ca_atom = residue.get_atom('CA') # NOTE: for all-atom, this might need to change
        
        if ca_atom is not None:
            # Extract coordinates
            ca_coords.append((ca_atom.x, ca_atom.y, ca_atom.z))
            
            # Get amino acid single-letter code
            sequence += residue.code
    
    return ca_coords, sequence
