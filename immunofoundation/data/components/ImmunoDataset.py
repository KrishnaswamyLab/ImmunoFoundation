import pandas as pd
import torch
from torch.utils.data import Dataset

from immunofoundation.data.components.preprocess_pdb import extract_ca_and_sequence
from immunofoundation.data.components.preprocess import extract_biochemical_properties

class ImmunoDataset(Dataset):
    def __init__(self, data_cfg, is_training):
        #TODO implement the is_training flag logic
        self.config = data_cfg
        self.csv_df = pd.read_csv(data_cfg.csv_file_path)
        self._init_metadata()

    def _init_metadata(self):
        pass

    def _process_csv_row(self,csv_row):
        '''
            returns: final_features: Dict containing all the information necessary for the model to train
        '''
        ca_coords_peptide, sequence_peptide, ca_coords_mhc, sequence_mhc = extract_ca_and_sequence(csv_row['pdb_path'])
        biochemical_properties = extract_biochemical_properties(csv_row)

        final_features = {}
        final_features['peptide_coordinates'] = ca_coords_peptide
        final_features['MHC_coordinates'] = ca_coords_mhc
        final_features['peptide_sequence'] = sequence_peptide
        final_features['MHC_sequence'] = sequence_mhc
        final_features['biochemical_properties'] = biochemical_properties

        return final_features

    def __getitem__(self, idx):

        csv_row = self.csv.iloc[idx]
        final_features = self._process_csv_row(csv_row) # get the features for this instance
        return final_features
    
    def __len__(self):
        return len(self.csv)
    