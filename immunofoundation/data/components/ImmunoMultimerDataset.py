import pandas as pd
import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph

from torch.utils.data import Dataset
import torch.nn.functional as F

from immunofoundation.data.components.preprocess_pdb import extract_ca_and_sequence
from immunofoundation.data.components.preprocess import extract_biochemical_properties
import warnings
warnings.filterwarnings("ignore")

# TODO: define amino acids to indices map as constant and use within __getitem__

class ImmunoMultimerDataset(Dataset):
    def __init__(self, data_cfg, is_training):
        self.data_cfg = data_cfg
        self.is_training = is_training
        self._init_metadata()

    def _init_metadata(self):
        pdb_csv = pd.read_csv(self.data_cfg.csv_path)
        self.raw_csv = pdb_csv
        num_records_80 = int(pdb_csv.shape[0]*self.data_cfg.train_size)-1

        if self.is_training:
            pdb_csv = pdb_csv.iloc[:num_records_80, :]
            self.csv = pdb_csv
            print (f"Training: {len(self.csv)} samples")
        else:
            pdb_csv = pdb_csv.iloc[num_records_80:, :]
            self.csv = pdb_csv
            print (f"Validation: {len(self.csv)} samples")

    def _process_csv_row(self, csv_row):
        '''
            returns: final_features: Dict containing all the information necessary for the model to train
        '''
        ca_coords_peptide, sequence_peptide, ca_coords_mhc, sequence_mhc = extract_ca_and_sequence(csv_row['cif_path'])
        biochemical_properties = extract_biochemical_properties(csv_row)

        final_features = {}
        final_features['peptide_len'] = len(sequence_peptide)
        final_features['mhc_len'] = len(sequence_mhc)
        final_features['peptide_coords'] = torch.tensor(ca_coords_peptide).float()
        final_features['mhc_coords'] = torch.tensor(ca_coords_mhc).float()
        final_features['peptide_sequence'] = sequence_peptide
        final_features['mhc_sequence'] = sequence_mhc
        final_features['biochemical_properties'] = torch.from_numpy(biochemical_properties)
        if self.data_cfg.structure.adj:
            final_features['peptide_adj'] = kneighbors_graph(ca_coords_peptide, n_neighbors = self.data_cfg.structure.k)
            final_features['mhc_adj'] = kneighbors_graph(ca_coords_mhc, n_neighbors = self.data_cfg.structure.k)
        else:
            final_features['peptide_adj'] = None
            final_features['mhc_adj'] = None
        peptide_distances = torch.cdist(final_features['peptide_coords'], final_features['peptide_coords'], 2)
        mhc_distances = torch.cdist(final_features['mhc_coords'], final_features['mhc_coords'], 2)
        final_features['peptide_mask'] = self.mask_residues((peptide_distances < self.data_cfg.mask.max_distance).sum(1) < self.data_cfg.mask.max_neighbors)
        final_features['mhc_mask'] = self.mask_residues((mhc_distances < self.data_cfg.mask.max_distance).sum(1) < self.data_cfg.mask.max_neighbors)
        return final_features

    def __getitem__(self, idx):
        csv_row = self.csv.iloc[idx]
        final_features = self._process_csv_row(csv_row) # get the features for this instance
        return final_features
    
    def __len__(self):
        return len(self.csv)

    def mask_residues(self, x):
        true_indices = torch.nonzero(x, as_tuple=True)[0]
        num_to_flip = int(len(true_indices) * self.data_cfg.mask.mask_rate)
        flip_indices = true_indices[torch.randperm(len(true_indices))[:num_to_flip]]
        x[flip_indices] = False
        return x.long()

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

def pad_square(x, max_len, use_torch=False, reverse=False):
    """
    Pads a 2D array (matrix) to shape (max_len, max_len) by adding zeros
    to the right and bottom (or left/top if reverse=True).

    Args:
        x: numpy or torch array of shape (H, W)
        max_len: int, desired final square size
        use_torch: whether to use torch padding
        reverse: if True, pad on the left/top instead of right/bottom

    Returns:
        Padded array of shape (max_len, max_len)
    """
    h, w = x.shape[:2]
    pad_h = max_len - h
    pad_w = max_len - w

    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Cannot pad: current ({h},{w}) > max_len {max_len}")

    if reverse:
        pad_spec = ((pad_h, 0), (pad_w, 0))
    else:
        pad_spec = ((0, pad_h), (0, pad_w))

    if use_torch:
        # torch.pad uses reversed order: (left, right, top, bottom)
        # For 2D tensor, flatten spec accordingly:
        pad = (pad_spec[1][0], pad_spec[1][1], pad_spec[0][0], pad_spec[0][1])
        return torch.nn.functional.pad(x, pad)
    else:
        return np.pad(x, pad_spec)
def custom_collate_multi(batch_list):
    """
    `batch_list` is a list of dict containing:
    - peptide coords [N_pep_res, 3]
    - MHC coords [N_mhc_res, 3]
    """
    max_len_peptide = max([x['peptide_len'] for x in batch_list])
    # max_len_peptide = max(list(map(lambda len(x['peptide_len']) : x, batch_list)))
    padded_peptide_ca_coords = torch.utils.data.default_collate([pad(rec['peptide_coords'], max_len=max_len_peptide) for rec in batch_list])
    mhc_ca_coords = torch.utils.data.default_collate([rec['mhc_coords'] for rec in batch_list])
    if batch_list[0]['peptide_adj'] is not None:
        peptide_adjs = torch.utils.data.default_collate([pad_square(rec['peptide_adj'], max_len=max_len_peptide) for rec in batch_list])
        mhc_adjs = torch.utils.data.default_collate([pad_square(rec['mhc_adj'], max_len=max_len_peptide) for rec in batch_list])
    else:
        peptide_adjs = torch.utils.data.default_collate([0]*len(batch_list))
        mhc_adjs = torch.utils.data.default_collate([0]*len(batch_list))
    biochemical_properties = torch.utils.data.default_collate([torch.tensor(rec['biochemical_properties']).float() for rec in batch_list])
    peptide_masks = torch.utils.data.default_collate([torch.tensor(rec['peptide_mask']).float() for rec in batch_list])
    mhc_masks = torch.utils.data.default_collate([torch.tensor(rec['mhc_mask']).float() for rec in batch_list])
    mhc_sequences = torch.utils.data.default_collate([rec['mhc_sequence'] for rec in batch_list])
    peptide_sequences = torch.utils.data.default_collate([rec['peptide_sequence'] for rec in batch_list])
    # TODO: include integer amino acid indices
    return {
        "mhc_coords": mhc_ca_coords,
        "peptide_coords": padded_peptide_ca_coords,
        "peptide_adjs": peptide_adjs,
        "mhc_adjs": mhc_adjs,
        "biochemical_properties": biochemical_properties,
        "mhc_sequence": mhc_sequences,
        "peptide_sequence": peptide_sequences,
        "mhc_masks": mhc_masks,
        "peptide_masks": peptide_masks
    }