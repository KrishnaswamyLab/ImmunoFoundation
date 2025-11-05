import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

from immunofoundation.data.components.preprocess_pdb import extract_ca_and_sequence
from immunofoundation.data.components.preprocess import extract_biochemical_properties
import warnings
warnings.filterwarnings("ignore")

# TODO: define amino acids to indices map as constant and use within __getitem__

class ImmunoDataset(Dataset):
    def __init__(self, data_cfg, is_training):
        self.data_cfg = data_cfg
        self.is_training = is_training
        self._init_metadata()

    def _init_metadata(self):
        pdb_csv = pd.read_csv(self.data_cfg.csv_path)
        self.raw_csv = pdb_csv

        if self.is_training:
            # get first 80% of samples for training
            num_records_80 = int(pdb_csv.shape[1]*0.8)-1
            pdb_csv = pdb_csv.iloc[:num_records_80, :]
            self.csv = pdb_csv
            print (f"Training: {len(self.csv)} samples")
        else:
            # get remaining 20%% of samples for val/test
            num_records_80 = int(pdb_csv.shape[1]*0.8)-1
            pdb_csv = pdb_csv.iloc[num_records_80:, :]
            self.csv = pdb_csv
            print (f"Validation: {len(self.csv)} samples")

    def _process_csv_row(self, csv_row):
        '''
            returns: final_features: Dict containing all the information necessary for the model to train
        '''
        ca_coords_peptide, sequence_peptide, ca_coords_mhc, sequence_mhc = extract_ca_and_sequence(csv_row['pdb_path'])
        biochemical_properties = extract_biochemical_properties(csv_row)

        final_features = {}
        final_features['peptide_len'] = len(sequence_peptide)
        final_features['mhc_len'] = len(sequence_mhc)
        final_features['peptide_coords'] = torch.tensor(ca_coords_peptide).float()
        final_features['mhc_coords'] = torch.tensor(ca_coords_mhc).float()
        final_features['peptide_sequence'] = sequence_peptide
        final_features['mhc_sequence'] = sequence_mhc
        final_features['biochemical_properties'] = torch.from_numpy(biochemical_properties)

        return final_features

    def __getitem__(self, idx):
        csv_row = self.csv.iloc[idx]
        final_features = self._process_csv_row(csv_row) # get the features for this instance
        return final_features
    
    def __len__(self):
        return len(self.csv)

def pad(x: np.ndarray, max_len: int, pad_idx=0, use_torch=False, reverse=False):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        use_torch: use torch padding method instead of numpy.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f"Invalid pad amount {pad_amt}")
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        return torch.pad(x, pad_widths)
    return np.pad(x, pad_widths)

def custom_collate(batch_list):
    """
    `batch_list` is a list of dict containing:
    - peptide coords [N_pep_res, 3]
    - MHC coords [N_mhc_res, 3]
    """
    max_len_peptide = max([x['peptide_len'] for x in batch_list])
    # max_len_peptide = max(list(map(lambda len(x['peptide_len']) : x, batch_list)))
    padded_peptide_ca_coords = torch.utils.data.default_collate([pad(rec['peptide_coords'], max_len=max_len_peptide) for rec in batch_list])
    mhc_ca_coords = torch.utils.data.default_collate([rec['mhc_coords'] for rec in batch_list])
    biochemical_properties = torch.utils.data.default_collate([torch.tensor(rec['biochemical_properties']).float() for rec in batch_list])
    mhc_sequences = torch.utils.data.default_collate([rec['mhc_sequence'] for rec in batch_list])
    peptide_sequences = torch.utils.data.default_collate([rec['peptide_sequence'] for rec in batch_list])
    # TODO: include integer amino acid indices
    return {
        "mhc_coords": mhc_ca_coords,
        "peptide_coords": padded_peptide_ca_coords,
        "biochemical_properties": biochemical_properties,
        "mhc_sequence": mhc_sequences,
        "peptide_sequence": peptide_sequences
    }