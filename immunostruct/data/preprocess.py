from tqdm import tqdm
import os
import torch
import numpy as np
import pandas as pd
from .utils import get_hash, pad_graph, to_dgl, pad_peptide_sequence, one_hot_encode_sequence

__all__ = [
    "preprocess_graphs", "preprocess_properties", "preprocess_properties_cancer_wt",
    "preprocess_hla", "preprocess_sequence_graph", "preprocess_sequence_graph_cancer_wt",
    "preprocess_sequence_graph_clinical",
    "preprocess_graph", "preprocess_sequence",
]

def preprocess_graphs(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]

    # Initialize an empty list to store the graphs
    graphs = []

    # Loop through the files and load each graph, showing a progress bar
    for file in tqdm(files, desc="Loading graphs"):
        file_path = os.path.join(directory, file)
        graph = torch.load(file_path)
        graphs.append(graph)

    print(f"Loaded {len(graphs)} graphs.")

    graphs = [x for x in graphs if ('NXVPMVATV' not in x.name) and ('X' not in x.name)]

    new_graphs = []
    names = set()

    for graph in graphs:
        if graph.name.split("Immuno")[1] not in names:
            names.add(graph.name.split("Immuno")[1])
            new_graphs.append(graph)

    #cut off h-bonding features for now
    for data in new_graphs:  # Assuming data_list is the list containing your graph data
        data.x = data.x[:, :-2]

    return new_graphs

def preprocess_properties(table, cancer=False):
    expanded_df = pd.read_table(table)

    if cancer:
        # used for cancer dataset
        expanded_df = expanded_df.dropna(subset='foreign')
        expanded_df[['allele1', 'allele2']] = expanded_df['allele'].str.split("-", expand=True)
        allele = expanded_df['allele1'] + "-" + (expanded_df['allele2'].str)[0] + "*" + (expanded_df['allele2'].str)[1:3] + ":" + (expanded_df['allele2'].str)[3:]
        expanded_df['pep_pair'] = expanded_df['mut_pep'] + allele
    else:
        # used for IEDB dataset
        expanded_df = expanded_df.dropna(subset='Foreignness_Score')
        expanded_df['pep_pair'] = expanded_df['peptide'] + expanded_df['allele']

    f_dict = dict(zip(expanded_df['pep_pair'], expanded_df['smoothed_foreign']))
    fp2_dict = dict(zip(expanded_df['pep_pair'], zip(expanded_df['Mprop1'], expanded_df['Mprop2'])))
    new_imm_dict = dict(zip(expanded_df['pep_pair'], expanded_df['immunogenicity']))

    expanded_pep_pair = expanded_df['pep_pair'].tolist()
    return f_dict, fp2_dict, new_imm_dict, expanded_pep_pair


def preprocess_properties_cancer_wt(table_cancer, table_wt):
    expanded_df_cancer = pd.read_table(table_cancer)
    expanded_df_wt = pd.read_table(table_wt)

    expanded_df_cancer = expanded_df_cancer.dropna(subset='foreign')
    expanded_df_cancer[['allele1', 'allele2']] = expanded_df_cancer['allele'].str.split("-", expand=True)
    allele = expanded_df_cancer['allele1'] + "-" + (expanded_df_cancer['allele2'].str)[0] + "*" + (expanded_df_cancer['allele2'].str)[1:3] + ":" + (expanded_df_cancer['allele2'].str)[3:]
    expanded_df_cancer['pep_pair_cancer'] = expanded_df_cancer['mut_pep'] + allele

    expanded_df_wt = expanded_df_wt.dropna(subset='foreign')
    expanded_df_wt[['allele1', 'allele2']] = expanded_df_wt['allele'].str.split("-", expand=True)
    allele = expanded_df_wt['allele1'] + "-" + (expanded_df_wt['allele2'].str)[0] + "*" + (expanded_df_wt['allele2'].str)[1:3] + ":" + (expanded_df_cancer['allele2'].str)[3:]
    expanded_df_wt['pep_pair_wt'] = expanded_df_wt['wt_pep'] + allele

    short_df_cancer = expanded_df_cancer[['mut_pep', 'wt_pep', 'allele', 'immunogenicity', 'pep_pair_cancer', 'smoothed_foreign', 'Mprop1', 'Mprop2']]
    short_df_wt = expanded_df_wt[['mut_pep', 'wt_pep', 'allele', 'immunogenicity', 'foreign', 'pep_pair_wt', 'Mprop1_wt', 'Mprop2_wt']]
    short_df_cancer = __dedup_property_df(short_df_cancer)
    short_df_wt = __dedup_property_df(short_df_wt)

    combined_df = pd.merge(short_df_cancer, short_df_wt, on=['mut_pep', 'wt_pep', 'allele', 'immunogenicity'])
    combined_df = combined_df[['mut_pep', 'wt_pep', 'allele', 'immunogenicity', 'pep_pair_cancer', 'pep_pair_wt', 'smoothed_foreign', 'Mprop1', 'Mprop1_wt', 'Mprop2', 'Mprop2_wt']]
    assert len(short_df_cancer) == len(short_df_wt) == len(combined_df)

    return combined_df

def __dedup_property_df(df):
    """
    In a few cases, there may be duplicates in entries (likely from different patients)
    that share the same ('mut_pep', 'wt_pep', 'allele') but have slightly different ('smoothed_foreign', 'Mprop1', 'Mprop2').
    We will deduplicate, keeping the entry with the highest foreignness if immunogenic and the lowest foreignness otherwise.
    """

    assert len(np.unique([str(item) for item in df[['mut_pep', 'wt_pep', 'allele', 'immunogenicity']].values])) \
        == len(np.unique([str(item) for item in df[['mut_pep', 'wt_pep', 'allele']].values])), \
            "`__dedup_property_df`: same ('mut_pep', 'wt_pep', 'allele') but different immunogenicity!"

    tuple_list = [str(item) for item in df[['mut_pep', 'wt_pep', 'allele']].values]
    duplicate_items = []
    duplicate_rows_list = []
    for item in tuple_list:
        if tuple_list.count(item) > 1:
            if item not in duplicate_items:
                duplicate_items.append(item)
                duplicate_rows_list.append([i for i, x in enumerate(tuple_list) if x == item])

    rows_to_drop = []
    for duplicate_rows in duplicate_rows_list:
        immunogenicity_arr = df.loc[duplicate_rows]['immunogenicity'].values
        assert len(np.unique(immunogenicity_arr)) == 1, \
            "`__dedup_property_df`: same ('mut_pep', 'wt_pep', 'allele') but different immunogenicity!"
        immunogenicity = immunogenicity_arr[0]
        foreign_key = 'smoothed_foreign' if 'smoothed_foreign' in df else 'foreign'
        foreignness_arr = df.loc[duplicate_rows][foreign_key].values
        if immunogenicity == 1:
            idx_to_keep = duplicate_rows[foreignness_arr.argmax()]
        else:
            assert immunogenicity == 0
            idx_to_keep = duplicate_rows[foreignness_arr.argmin()]
        rows_to_drop.extend(list(set(duplicate_rows) - set([idx_to_keep])))

    if len(rows_to_drop) > 0:
        df = df.drop(index=rows_to_drop)

    return df

def preprocess_hla(expanded_pep_pair, hla_path):
    hla_df = pd.read_csv(hla_path)
    hla_dict_true = dict(zip(hla_df['allele'], hla_df['seqs']))

    name_mapper = {}

    for seq in expanded_pep_pair:
        pep, hla = seq.split("HLA-")
        unfolded = hla_dict_true["HLA-"+hla]
        name = unfolded + pep
        hashed = get_hash(name)[:5]
        name_mapper[seq] = (name, name[-99:]+"_"+hashed, pep)

    return name_mapper

def preprocess_sequence_graph(name_mapper, new_graphs, new_imm_dict, f_dict):
    strings = [x.name.split("Immuno")[1] for x in new_graphs]
    names = set(strings)
    to_remove = []
    for x, y in name_mapper.items():
        if y[1] not in names:
            to_remove.append(x)

    for i in to_remove:
        del name_mapper[i]

    print("new sequence table size: {}, removed {}".format(len(name_mapper), len(to_remove)))

    # table -> graph
    to_remove = set()
    mapper_names = set(y[1] for x, y in name_mapper.items())

    for i in strings:
        if i not in mapper_names:
            to_remove.add(i)

    new_graphs = [x for x in new_graphs if x.name.split("Immuno")[1] not in to_remove]
    strings = [x for x in new_graphs]

    graph_mapper = {x.name.split("Immuno")[1]: x for x in new_graphs}

    print("new graph list size: {}, removed {}".format(len(strings), len(to_remove)))

    for x, y in name_mapper.items():
        immuno_score = new_imm_dict[x]
        f_score = f_dict[x]
        graph = graph_mapper[y[1]]

        graph.y = torch.tensor([immuno_score, f_score], dtype=torch.float)  # We use a one-element tensor for each graph-level label
        
        src, dst = graph.edge_index
        edge_index_pyg = torch.stack([src, dst], dim=0)
        graph.internal_coords = cartesian_to_internal_coordinates(graph.coords, edge_index_pyg)
        
        graph.x = torch.cat([graph.x, graph.coords], dim=-1)

        graph.x = graph.x.to(dtype=torch.float32)
        graph.y = graph.y.to(dtype=torch.float32)

    return name_mapper, graph_mapper

def preprocess_sequence_graph_cancer_wt(combined_df, name_mapper_cancer, name_mapper_wt, graphs_cancer, graphs_wt):
    """
    `name_mapper_cancer` and `name_mapper_wt`:
        key: pep_pair
        value: (full sequence, truncated sequence and hash, peptide)
    """

    to_remove_cancer_all = set()
    to_remove_wt_all = set()

    strings_cancer = [x.name.split("Immuno")[1] for x in graphs_cancer]
    to_remove_cancer = set()
    for k, v in name_mapper_cancer.items():
        if v[1] not in set(strings_cancer):
            to_remove_cancer.add(k)
            to_remove_cancer_all.add(k)
    for k in to_remove_cancer:
        del name_mapper_cancer[k]
    print("(Cancer) new sequence table size: {}, removed {}".format(len(name_mapper_cancer), len(to_remove_cancer)))

    strings_wt = [x.name.split("Immuno")[1] for x in graphs_wt]
    to_remove_wt = set()
    for k, v in name_mapper_wt.items():
        if v[1] not in set(strings_wt):
            to_remove_wt.add(k)
            to_remove_wt_all.add(k)
    for k in to_remove_wt:
        del name_mapper_wt[k]
    print("(WT) new sequence table size: {}, removed {}".format(len(name_mapper_wt), len(to_remove_wt)))

    # Table -> graph
    to_remove_cancer = set()
    for k in strings_cancer:
        if k not in set(v[1] for _, v in name_mapper_cancer.items()):
            to_remove_cancer.add(k)
            to_remove_cancer_all.add(k)
    for k in to_remove_cancer:
        del name_mapper_cancer[k]
    print("(Cancer) new graph list size: {}, removed {}".format(len(name_mapper_cancer), len(to_remove_cancer)))

    to_remove_wt = set()
    for k in strings_wt:
        if k not in set(v[1] for _, v in name_mapper_wt.items()):
            to_remove_wt.add(k)
            to_remove_wt_all.add(k)
    for k in to_remove_wt:
        del name_mapper_wt[k]
    print("(WT) new graph list size: {}, removed {}".format(len(name_mapper_wt), len(to_remove_wt)))

    # Cross check cancer vs. wt and remove unmatched sequences and graphs.
    cancer_wt_mapper = dict(zip(combined_df['pep_pair_cancer'], combined_df['pep_pair_wt']))
    wt_cancer_mapper = dict(zip(combined_df['pep_pair_wt'], combined_df['pep_pair_cancer']))

    to_remove_cancer = set()
    for k, v in name_mapper_cancer.items():
        k_wt = cancer_wt_mapper[k]
        if k_wt not in name_mapper_wt.keys():
            to_remove_cancer.add(k)
            to_remove_cancer_all.add(k)
    for k in to_remove_cancer:
        del name_mapper_cancer[k]

    to_remove_wt = set()
    for k, v in name_mapper_wt.items():
        k_cancer = wt_cancer_mapper[k]
        if k_cancer not in name_mapper_cancer.keys():
            to_remove_wt.add(k)
            to_remove_wt_all.add(k)
    for k in to_remove_wt:
        del name_mapper_wt[k]

    print("After cross-checking (cancer vs. wt), final list size: {}, removed {} from cancer and {} from wt".format(
        len(name_mapper_cancer), len(to_remove_cancer), len(to_remove_wt)))

    # Remove corresponding rows from `combinded_df`.
    for k in to_remove_cancer_all:
        combined_df = combined_df[combined_df['pep_pair_cancer'] != k]
    for k in to_remove_wt_all:
        combined_df = combined_df[combined_df['pep_pair_wt'] != k]

    # Organize the graph dicts.
    graphs_cancer = [item for item in graphs_cancer if item.name.split("Immuno")[1] not in to_remove_cancer_all]
    graphs_wt = [item for item in graphs_wt if item.name.split("Immuno")[1] not in to_remove_wt_all]
    graph_mapper_cancer = {item.name.split("Immuno")[1]: item for item in graphs_cancer}
    graph_mapper_wt = {item.name.split("Immuno")[1]: item for item in graphs_wt}

    for k, v in name_mapper_cancer.items():
        k_wt = cancer_wt_mapper[k]
        v_wt = name_mapper_wt[k_wt]

        df_entry = combined_df[np.logical_and(combined_df['pep_pair_cancer'] == k, combined_df['pep_pair_wt'] == k_wt)]
        assert len(df_entry) == 1
        immuno_score = df_entry['immunogenicity'].item()
        foreignness_score = df_entry['smoothed_foreign'].item()

        graph_cancer = graph_mapper_cancer[v[1]]
        graph_cancer.x = torch.cat([graph_cancer.x, graph_cancer.coords], dim=-1)
        graph_cancer.y = torch.tensor([immuno_score, foreignness_score], dtype=torch.float)  # We use a one-element tensor for each graph-level label
        graph_cancer.x = graph_cancer.x.to(dtype=torch.float32)
        graph_cancer.y = graph_cancer.y.to(dtype=torch.float32)
        
        # Generate internal coordinates
        src, dst = graph_cancer.edge_index
        edge_index_pyg = torch.stack([src, dst], dim=0)
        graph_cancer.internal_coords = cartesian_to_internal_coordinates(graph_cancer.coords, edge_index_pyg)

        graph_wt = graph_mapper_wt[v_wt[1]]
        if graph_wt.x.shape[1] < graph_cancer.x.shape[1]:
            graph_wt.x = torch.cat([graph_wt.x, graph_wt.coords], dim=-1)
            graph_wt.y = torch.tensor([0, combined_df['smoothed_foreign'].min()], dtype=torch.float)  # We use a one-element tensor for each graph-level label
            graph_wt.x = graph_wt.x.to(dtype=torch.float32)
            graph_wt.y = graph_wt.y.to(dtype=torch.float32)
            
            # Generate internal coordinates
            src, dst = graph_wt.edge_index
            edge_index_pyg = torch.stack([src, dst], dim=0)
            graph_wt.internal_coords = cartesian_to_internal_coordinates(graph_wt.coords, edge_index_pyg)
            
            assert graph_wt.x.shape[1] == graph_cancer.x.shape[1]
        else:
            # In this case, the same graph has already been iterated. Move on.
            assert graph_wt.x.shape[1] == graph_cancer.x.shape[1]

    return combined_df, name_mapper_cancer, name_mapper_wt, graph_mapper_cancer, graph_mapper_wt

def preprocess_sequence_graph_clinical(graph_directory, seq_path):
    graphs = preprocess_graphs(graph_directory)

    seq_df = pd.read_table(seq_path)
    name_mapper = {}
    for _, row in seq_df.iterrows():
        pep = row['mut_pep']
        seq = row['combo']
        unfolded = row['hla_seq']
        name = unfolded + pep
        hashed = get_hash(name)[:5]
        name_mapper[seq] = (name, name[-99:]+"_"+hashed, pep)

    strings = [x.name.split("Immuno")[1] for x in graphs]
    names = set(strings)
    to_remove = []
    for x, y in name_mapper.items():
        if y[1] not in names:
            to_remove.append(x)
    for i in to_remove:
        del name_mapper[i]
    print("new sequence table size: {}, removed {}".format(len(name_mapper), len(to_remove)))
    # table -> graph
    to_remove = set()
    mapper_names = set(y[1] for x, y in name_mapper.items())
    for i in strings:
        if i not in mapper_names:
            to_remove.add(i)
    graphs = [x for x in graphs if x.name.split("Immuno")[1] not in to_remove]
    strings = [x for x in graphs]
    graph_mapper = {x.name.split("Immuno")[1]: x for x in graphs}
    print("new graph list size: {}, removed {}".format(len(strings), len(to_remove)))

    for x, y in name_mapper.items():
        graph = graph_mapper[y[1]]
        graph.x = torch.cat([graph.x, graph.coords], dim=-1)
        graph.x = graph.x.to(dtype=torch.float32)
        
        # Generate internal coordinates for consistency with other datasets
        src, dst = graph.edge_index
        edge_index_pyg = torch.stack([src, dst], dim=0)
        graph.internal_coords = cartesian_to_internal_coordinates(graph.coords, edge_index_pyg)

    return name_mapper, graph_mapper


def preprocess_graph(graph_mapper, feature_size, coord_size):
    max_nodes = max(graph.num_nodes for graph in graph_mapper.values())

    padded_graphs = {name: pad_graph(graph, max_nodes, feature_size, coord_size) for name, graph in graph_mapper.items()}

    graph_mapper = {name: to_dgl(graph) for name, graph in padded_graphs.items()}
    return graph_mapper

def preprocess_sequence(name_mapper, amino_acids, padding_char):

    max_full_length = max(len(y[0]) for x, y in name_mapper.items())
    max_pep_length = max(len(y[2]) for x, y in name_mapper.items())

    name_mapper = {x:(pad_peptide_sequence(a, max_full_length, padding_char),
                      b, pad_peptide_sequence(c, max_pep_length, padding_char)) for x, (a, b, c) in name_mapper.items()}

    encoded_full_sequence_map = {x:one_hot_encode_sequence(a, amino_acids, padding_char) for x, (a, b, c) in name_mapper.items()}
    encoded_peptide_map = {x:one_hot_encode_sequence(c, amino_acids, padding_char) for x, (a, b, c) in name_mapper.items()}

    return encoded_full_sequence_map, encoded_peptide_map

def preprocess_hla_old(table, graphs, fp2_dict, f_dict, new_imm_dict):
    hla_df = pd.read_csv(table)
    hla_dict_true = dict(zip(hla_df['allele'], hla_df['seqs']))

    new_values_fp2_values = []
    new_values_f_values = []

    new_imm_values = []
    peptide_order = []

    hla_name = []
    new_graphs = []

    for graph in graphs:
        string = graph.name
        start = string.find("LPKPLTLR")
        if start != -1:  # Check if the substring exists in the string
            end = string.find("_", start)
            if end != -1:
                substring = string[start:end]
                new_name = substring[8:]
                if new_name not in fp2_dict:
                    continue
                new_values_fp2_values.append(fp2_dict[new_name])
                new_values_f_values.append(f_dict[new_name])
                peptide_order.append(new_name)
                new_imm_values.append(new_imm_dict[new_name])

                start_index = 24
                end_sequence = "LPKPLTLR"
                end_index = graph.name.find(end_sequence, start_index) + len(end_sequence)
                sub_hla = graph.name[start_index:end_index] if end_index != -1 else ""
                hla_name.append(sub_hla)
                new_graphs.append(graph)

            else:
                print(f"Underscore not found in string: {string}")
        else:
            print(f"'LPKPLTLR' not found in string: {string}")

    # Update dictionary values with shorter strings
    for key, value in hla_dict_true.items():
        for short_string in hla_name:
            if short_string in value:
                hla_dict_true[key] = short_string
                break  # Stop checking after the first match is found

    inverted_hla_dict = {}
    for key, value in hla_dict_true.items():
        if value in inverted_hla_dict:
            inverted_hla_dict[value].append(key)
        else:
            inverted_hla_dict[value] = [key]

    corresponding_alleles = []

    # Loop through each string in hla_name
    for string in hla_name:
        string = string[2:]
        found = False
        for key in inverted_hla_dict:
            if string in key:
                corresponding_alleles.append(inverted_hla_dict[key])
                found = True
                break
        if not found:
            print(["Allele not found"])
            corresponding_alleles.append(["Allele not found"])

    for data, score in zip(new_graphs, new_values_f_values):
        data.y = torch.tensor([score], dtype=torch.float)  # We use a one-element tensor for each graph-level label
        data.x = torch.cat([data.x, data.coords], dim=-1)

        data.x = data.x.to(dtype=torch.float32)
        data.y = data.y.to(dtype=torch.float32)

    return new_graphs, corresponding_alleles, new_values_fp2_values, new_values_f_values, peptide_order

def cartesian_to_internal_coordinates(coord_feat, edge_index):
        # Convert to numpy
        coords_np = coord_feat.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy()
        
        num_nodes = coords_np.shape[0]
        internal_features = []
        
        # adjacency list for neighbor lookup
        adjacency = {}
        for i in range(num_nodes):
            adjacency[i] = []
        for src, dst in edge_index_np.T:
            adjacency[src].append(dst)
            adjacency[dst].append(src)
        
        for node_i in range(num_nodes):
            node_features = []
            neighbors = adjacency[node_i]
            
            if len(neighbors) == 0:
                # isolated node - use zero features
                node_features = [0.0] * 6  # 6 internal coordinate features
            else:
                # bond lengths to all neighbors
                distances = [];
                for neighbor in neighbors[:3]:
                    dist = np.linalg.norm(coords_np[node_i] - coords_np[neighbor])
                    distances.append(dist)
                    
                # pad to 3 distances
                while len(distances) < 3:
                    distances.append(0.0)
                node_features.extend(distances[:3])
                
                # bond angles (if at least 2 neighbors)
                if len(neighbors) >= 2:
                    for i in range(min(2, len(neighbors)-1)):
                        v1 = coords_np[neighbors[i]] - coords_np[node_i]
                        v2 = coords_np[neighbors[i+1]] - coords_np[node_i]
                        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if norm1 > 1e-8 and norm2 > 1e-8:  # Check for zero vectors
                            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                            angle = np.arccos(np.clip(cos_angle, -1, 1))
                            node_features.append(angle)
                        else:
                            node_features.append(0.0)
                else:
                    node_features.append(0.0)
                
                # dihedral angles (if at least 3 neighbors forming a chain)
                if len(neighbors) >= 3:
                    try:
                        v1 = coords_np[neighbors[0]] - coords_np[node_i]
                        v2 = coords_np[neighbors[1]] - coords_np[neighbors[0]]
                        v3 = coords_np[neighbors[2]] - coords_np[neighbors[1]]
                        
                        # Check for zero vectors
                        if (np.linalg.norm(v1) > 1e-8 and 
                            np.linalg.norm(v2) > 1e-8 and 
                            np.linalg.norm(v3) > 1e-8):
                            
                            n1 = np.cross(v1, v2)
                            n2 = np.cross(v2, v3)
                            
                            # Check for zero normals (collinear vectors)
                            if np.linalg.norm(n1) > 1e-8 and np.linalg.norm(n2) > 1e-8:
                                v2_norm = v2 / np.linalg.norm(v2)
                                dihedral = np.arctan2(
                                    np.dot(np.cross(n1, n2), v2_norm), 
                                    np.dot(n1, n2)
                                )
                                node_features.append(dihedral)
                            else:
                                node_features.append(0.0)
                        else:
                            node_features.append(0.0)
                    except:
                        node_features.append(0.0)
                else:
                    node_features.append(0.0)
                
                # Pad to 6 features
                while len(node_features) < 6:
                    node_features.append(0.0)
            
            internal_features.append(node_features[:6])
            
        return torch.tensor(internal_features, dtype=torch.float32, device=coord_feat.device)