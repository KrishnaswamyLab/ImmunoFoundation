
import os
from torch.utils.data import Dataset
import random
import torch
import argparse
import numpy as np
from immunofoundation.data.components.preprocess import preprocess_graphs, preprocess_graph, preprocess_sequence, preprocess_sequence_graph, preprocess_properties, preprocess_hla
from .utils import duplicate_check, RandomRotation, one_hot_encode_sequence
from collections import Counter

class ImmunoDataset(Dataset):
    def __init__(self, data_cfg, is_training):
        #TODO implement the is_training flag logic
        self.config = data_cfg
        self.graph_directory = data_cfg.graphs_path
        # self.property_path = property_path
        # self.hla_path = hla_path

        # self.sequence_pad_count = config.sequence_pad_count
        # self.structure_pad_count = config.structure_pad_count

        graphs = preprocess_graphs(self.graph_directory)
        # f_dict, fp2_dict, new_imm_dict, expanded_pep_pair = preprocess_properties(property_path, True if "Cancer" in graph_directory else False)
        # name_mapper = preprocess_hla(expanded_pep_pair, hla_path)
        # name_mapper, graph_mapper = preprocess_sequence_graph(name_mapper, graphs, new_imm_dict, f_dict)
        # graph_mapper = preprocess_graph(graph_mapper, config.feature_size, config.coord_size)
        # encoded_full_sequence_map, encoded_peptide_map = preprocess_sequence(name_mapper, AMINO_ACIDS, PADDING_CHAR)

        # self.organize(name_mapper, encoded_full_sequence_map, encoded_peptide_map, fp2_dict, new_imm_dict, f_dict, graph_mapper)
        self.normalize()

    def organize(self, name_mapper, encoded_full_sequence_map, encoded_peptide_map, fp2_dict, new_imm_dict, f_dict, graph_mapper):
        names = [(x, a, b, c) for x, (a, b, c) in name_mapper.items()]

        encoded_full_sequence = [encoded_full_sequence_map[x[0]] for x in names]
        encoded_peptide_sequence = [encoded_peptide_map[x[0]] for x in names]

        protein_reg_values = [fp2_dict[x[0]] for x in names]
        protein_immuno_values = [new_imm_dict[x[0]] for x in names]

        class_weights = Counter(protein_immuno_values)
        self.class_weights = class_weights
        print(class_weights)

        protein_reg_values_f = [f_dict[x[0]] for x in names]

        dgl_filtered_graphs = [graph_mapper[x[2]] for x in names]

        duplicate_check(encoded_full_sequence, protein_reg_values, dgl_filtered_graphs)

        self.encoded_full_sequence = torch.tensor(np.array(encoded_full_sequence), dtype=torch.float32)
        self.encoded_peptide_sequence = torch.tensor(np.array(encoded_peptide_sequence), dtype=torch.float32)
        self.regression_values = torch.tensor(np.array(protein_reg_values), dtype=torch.float32)
        self.binary_values = torch.tensor(np.array(protein_immuno_values), dtype=torch.float32)
        self.regression_values_f = torch.tensor(np.array(protein_reg_values_f), dtype=torch.float32)

        self.graphs = dgl_filtered_graphs

        print("Preprocess Complete")

    def normalize(self):
        self.min = torch.min(self.regression_values_f)
        self.max = torch.max(self.regression_values_f)
        self.regression_values_f = 2 * (self.regression_values_f - (self.max+self.min)/2)/(self.max-self.min)

    def denormalize(self, output):
        return output / 2 * (self.max - self.min) + (self.max+self.min)/2

    def transform(self):
        return RandomRotation()
    
    def mask_sequence(self, full, peptide, padding_char):
        length = len(full)- len(peptide)

        inds = [i for i in range(length)]
        to_mask = random.sample(inds, self.sequence_pad_count)

        pad_one_hot = torch.tensor(np.array(one_hot_encode_sequence(padding_char, AMINO_ACIDS, PADDING_CHAR)), dtype=torch.float32)

        for i in to_mask:
            full[i] = pad_one_hot

        return full

    # perform this after self supervision on single structure
    def mask_structure(self, graph):
        inds = [i for i in range(len(graph.ndata['x']))]
        to_mask = random.sample(inds, self.structure_pad_count)

        for i in to_mask:
            if torch.sum(graph.ndata['x'][i,:-3]) > 1: # self supervision structure
                continue
            else:
                graph.ndata['x'][i,:-3] = torch.full(graph.ndata['x'][i,:-3].shape, 0)

        return graph

    def mask_single_structure(self, graph):
        inds = [i for i in range(len(graph.ndata['x']))]

        for _ in inds: # loop until we find a valid, non padded amino acid
            to_mask = random.choice(inds)
            amino_acid = torch.nonzero(graph.ndata['x'][to_mask,:-3], as_tuple=True)[0]
            if amino_acid.numel():
                graph.ndata['x'][to_mask,:-3] = torch.full(graph.ndata['x'][to_mask,:-3].shape, 1)
                return graph, amino_acid

        print("unmaskable graph: " , graph.ndata['x'])
        return graph, torch.tensor([0])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.encoded_full_sequence[idx], self.encoded_peptide_sequence[idx], self.regression_values[idx], self.binary_values[idx], self.regression_values_f[idx]