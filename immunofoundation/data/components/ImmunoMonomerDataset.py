import pandas as pd
import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph

from torch.utils.data import Dataset
import torch.nn.functional as F

from immunofoundation.data.components.preprocess_pdb import extract_ca_and_sequence, normalize_coords
from immunofoundation.data.components.preprocess import extract_biochemical_properties
import warnings
warnings.filterwarnings("ignore")

# TODO: define amino acids to indices map as constant and use within __getitem__

class ImmunoMonomerDataset(Dataset):
    def __init__(self, data_cfg, is_training):
        self.data_cfg = data_cfg
        self.is_training = is_training
        self._init_metadata()

    def _init_metadata(self):
        pdb_csv = pd.read_csv(self.data_cfg.csv_path)
        self.raw_csv = pdb_csv
        n = pdb_csv.shape[0]
        train_size = float(getattr(self.data_cfg, 'train_size', 1.0))
        if train_size >= 1.0 or train_size <= 0.0:
            # Use all data for both train and val if train_size is 1.0 or 0.0
            self.csv = pdb_csv
            print(f"Using all {len(self.csv)} samples (no split)")
        else:
            train_len = int(n * train_size)
            if self.is_training:
                self.csv = pdb_csv.iloc[:train_len, :]
                print(f"Training: {len(self.csv)} samples")
            else:
                self.csv = pdb_csv.iloc[train_len:, :]
                print(f"Validation: {len(self.csv)} samples")

    def _process_csv_row(self, csv_row):
        '''
            returns: final_features: Dict containing all the information necessary for the model to train
        '''
        raw_coords, sequence, _, _ = extract_ca_and_sequence(csv_row['cif_path'])

        final_features = {}
        final_features['len'] = len(sequence)
        final_features['sequence'] = sequence

        # Compute mask on raw Angstrom coordinates before normalization
        raw_coords_tensor = torch.tensor(raw_coords).float()
        distances = torch.cdist(raw_coords_tensor, raw_coords_tensor, 2)
        final_features['mask'] = self.mask_residues((distances < self.data_cfg.mask.max_distance).sum(1) < self.data_cfg.mask.max_neighbors)

        # Normalize coords and build adjacency after masking
        coords = normalize_coords(raw_coords)
        final_features['coords'] = torch.tensor(coords).float()
        if self.data_cfg.structure.adj:
            final_features['adj'] = kneighbors_graph(coords, n_neighbors = self.data_cfg.structure.k)
        else:
            final_features['adj'] = None
        # include label if present in CSV (optional for supervised finetuning)
        # default label column name expected: 'immunogenicity'
        if 'immunogenicity' in csv_row.index:
            try:
                final_features['label'] = int(csv_row['immunogenicity'])
            except Exception:
                # try mapping non-integer labels to 0/1
                if str(csv_row['immunogenicity']).lower() in ('true', '1', 'yes'):
                    final_features['label'] = 1
                else:
                    final_features['label'] = 0
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
        # For 2D tensor, flatten spec accordingly:
        pad = (pad_spec[1][0], pad_spec[1][1], pad_spec[0][0], pad_spec[0][1])
        # Handle scipy sparse matrices by converting to dense array first
        if hasattr(x, 'toarray'):
            x = x.toarray()
        return torch.nn.functional.pad(torch.tensor(x), pad)
    else:
        return np.pad(x, pad_spec)
def custom_collate_mono(batch_list):
    """
    `batch_list` is a list of dict containing:
    - coords [N_pep_res, 3]
    """
    max_len = max([x['len'] for x in batch_list])
    padded_coords = torch.utils.data.default_collate([pad(rec['coords'], max_len=max_len) for rec in batch_list])
    if batch_list[0]['adj'] is not None:
        adjs = torch.utils.data.default_collate([pad_square(rec['adj'], max_len=max_len, use_torch=True) for rec in batch_list])
    else:
        adjs = torch.utils.data.default_collate([0]*len(batch_list))
    masks = torch.utils.data.default_collate([pad(torch.tensor(rec['mask']).float(), max_len) for rec in batch_list])
    # keep sequences as python list so ESM wrapper can batch-convert them
    sequences = [rec['sequence'] for rec in batch_list]

    batch = {
        "coords": padded_coords,
        "adjs": adjs,
        "sequence": sequences,
        "masks": masks,
    }
    # optional fields: label and biochem
    if 'label' in batch_list[0]:
        labels = torch.tensor([rec['label'] for rec in batch_list], dtype=torch.long)
        batch['label'] = labels
    if 'biochem' in batch_list[0]:
        biochems = torch.utils.data.default_collate([rec['biochem'] for rec in batch_list])
        batch['biochem'] = biochems

    return batch