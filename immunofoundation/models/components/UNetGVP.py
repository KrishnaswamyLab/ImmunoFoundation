###########################################################################################
# UNet GVP Structure Model
# Adapted from: https://github.com/VirtualProteins/GNN_UNet
# Original class: UnetGVPGNNModel_Enc_Dec_Con
#
# Wraps the UNet GVP encoder to match the ImmunoFoundation StructureModel interface:
#   forward(adj, coords, node_features=None, **kwargs) -> (B, max_len, out_dim)
###########################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU
from torch_geometric.nn import fps, MLP, GINConv, knn_graph
from torch_scatter import scatter_add

from immunofoundation.models.components import gvp
from immunofoundation.models.components.radial import RadialEmbeddingBlock


class UNetGVPStructureModel(nn.Module):

    def __init__(self, struc_cfg):
        super().__init__()
        self.cfg = struc_cfg

        s_dim = struc_cfg.s_dim
        v_dim = struc_cfg.v_dim
        s_dim_edge = struc_cfg.s_dim_edge
        v_dim_edge = struc_cfg.v_dim_edge
        r_max = struc_cfg.r_max
        num_bessel = struc_cfg.num_bessel
        num_polynomial_cutoff = struc_cfg.num_polynomial_cutoff
        num_layers = struc_cfg.num_layers
        residual = struc_cfg.residual
        out_dim = struc_cfg.out_dim
        self.k = struc_cfg.k

        _DEFAULT_V_DIM = (s_dim, v_dim)
        _DEFAULT_E_DIM = (s_dim_edge, v_dim_edge)
        _DEFAULT_V_DIM_UP = (s_dim * 2, v_dim * 2)
        self.r_max = r_max
        self.num_layers = num_layers
        activations = (F.relu, None)

        # Node embedding: projects ESM features (or coords) to s_dim
        emb_in_dim = getattr(struc_cfg, 'esm_dim', 1280)
        self.emb_in = nn.Linear(emb_in_dim, s_dim)
        self.W_v = nn.Sequential(
            gvp.LayerNorm((s_dim, 0)),
            gvp.GVP(
                (s_dim, 0),
                _DEFAULT_V_DIM,
                activations=(None, None),
                vector_gate=True,
            ),
        )

        # Edge embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )

        self.num_pools = (num_layers + 1) // 2

        # Edge embedding layers for encoder path (one per pool level)
        self.W_e_d = ModuleList()
        for _ in range(self.num_pools):
            w_e = nn.Sequential(
                gvp.LayerNorm((self.radial_embedding.out_dim, 1)),
                gvp.GVP(
                    (self.radial_embedding.out_dim, 1),
                    _DEFAULT_E_DIM,
                    activations=(None, None),
                    vector_gate=True,
                ),
            )
            self.W_e_d.append(w_e)

        # Edge embedding layers for decoder path (one per pool level)
        self.W_e_up = ModuleList()
        for _ in range(self.num_pools):
            w_e = nn.Sequential(
                gvp.LayerNorm((self.radial_embedding.out_dim, 1)),
                gvp.GVP(
                    (self.radial_embedding.out_dim, 1),
                    _DEFAULT_E_DIM,
                    activations=(None, None),
                    vector_gate=True,
                ),
            )
            self.W_e_up.append(w_e)

        # Encoder GVP conv layers
        self.layers_d = nn.ModuleList(
            gvp.GVPConvLayer(
                _DEFAULT_V_DIM,
                _DEFAULT_E_DIM,
                activations=activations,
                vector_gate=True,
                residual=residual,
            )
            for _ in range(self.num_pools)
        )

        # Decoder GVP conv layers
        self.layers_up = nn.ModuleList(
            gvp.GVPConvLayer(
                _DEFAULT_V_DIM,
                _DEFAULT_E_DIM,
                activations=activations,
                vector_gate=True,
                residual=residual,
            )
            for _ in range(self.num_pools)
        )

        # Output GVP
        self.W_out = nn.Sequential(
            gvp.LayerNorm(_DEFAULT_V_DIM),
            gvp.GVP(
                _DEFAULT_V_DIM,
                (s_dim, 0),
                activations=activations,
                vector_gate=True,
            ),
        )

        # GINConv layers for pooling (encoder path)
        self.reds = ModuleList()
        for _ in range(self.num_pools):
            mlp = MLP([s_dim, s_dim, s_dim], act='relu', norm=None)
            self.reds.append(GINConv(nn=mlp, train_eps=False))

        # Skip connection projection layers (2*dim -> dim)
        self.lin_s = ModuleList()
        self.lin_v = ModuleList()
        for _ in range(2):
            self.lin_s.append(Linear(_DEFAULT_V_DIM_UP[0], _DEFAULT_V_DIM[0]))
            self.lin_v.append(Linear(_DEFAULT_V_DIM_UP[1], _DEFAULT_V_DIM[1]))
        self.act_lin = ReLU()

        # Final projection: s_dim -> out_dim
        self.final_proj = nn.Linear(s_dim, out_dim)

    def _compute_edge_features(self, pos, edge_index):
        """Compute edge vectors, distances, and radial+direction embeddings."""
        vectors = pos[edge_index[0]] - pos[edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        # Clamp lengths to avoid division by zero in Bessel basis (sin(w*x)/x)
        lengths_safe = lengths.clamp(min=1e-6)
        h_E = (
            self.radial_embedding(lengths_safe),
            torch.nan_to_num(torch.div(vectors, lengths_safe)).unsqueeze(-2),
        )
        return h_E

    def _dense_to_sparse(self, adj, coords, node_features=None):
        """
        Convert dense padded batch tensors to flat sparse PyG-style tensors.

        Args:
            adj: (B, max_len, max_len) dense adjacency
            coords: (B, max_len, 3) dense coordinates
            node_features: (B, max_len, F) optional node features

        Returns:
            pos: (total_nodes, 3)
            x: (total_nodes, F) or None
            edge_index: (2, total_edges)
            batch_vec: (total_nodes,)
            node_counts: (B,) tensor of actual node counts
        """
        B, max_len, _ = coords.shape
        device = coords.device

        all_pos = []
        all_x = []
        all_edges = []
        node_counts = []
        offset = 0

        for i in range(B):
            # Detect real (non-padded) nodes: rows with any nonzero entry in adj
            real_mask = adj[i].abs().sum(-1) > 0  # (max_len,)
            n_real = real_mask.sum().item()

            if n_real == 0:
                # Fallback: use coords to detect (in case adj is all zero)
                real_mask = coords[i].abs().sum(-1) > 0
                n_real = real_mask.sum().item()
                if n_real == 0:
                    n_real = 1  # at least one node
                    real_mask[0] = True

            node_counts.append(n_real)
            real_indices = torch.where(real_mask)[0]

            # Extract positions
            all_pos.append(coords[i, real_indices])

            # Extract node features
            if node_features is not None:
                all_x.append(node_features[i, real_indices])

            # Extract edges from adjacency: map local indices to global
            # Create a mapping from original indices to new compact indices
            idx_map = torch.full((max_len,), -1, dtype=torch.long, device=device)
            idx_map[real_indices] = torch.arange(n_real, device=device)

            # Get edges from adj
            sub_adj = adj[i][real_indices][:, real_indices]
            local_edges = sub_adj.nonzero(as_tuple=False).t()  # (2, n_edges)

            # Offset to global indices
            all_edges.append(local_edges + offset)
            offset += n_real

        pos = torch.cat(all_pos, dim=0)
        x = torch.cat(all_x, dim=0) if node_features is not None else None
        edge_index = torch.cat(all_edges, dim=1) if all_edges else torch.zeros(2, 0, dtype=torch.long, device=device)
        batch_vec = torch.cat([
            torch.full((nc,), i, dtype=torch.long, device=device)
            for i, nc in enumerate(node_counts)
        ])
        node_counts = torch.tensor(node_counts, dtype=torch.long, device=device)

        return pos, x, edge_index, batch_vec, node_counts

    def _sparse_to_dense(self, node_embeddings, batch_vec, node_counts, max_len):
        """
        Convert flat sparse node embeddings back to dense (B, max_len, D) tensor.

        Args:
            node_embeddings: (total_nodes, D)
            batch_vec: (total_nodes,)
            node_counts: (B,) tensor
            max_len: int

        Returns:
            dense: (B, max_len, D)
        """
        B = node_counts.shape[0]
        D = node_embeddings.shape[-1]
        device = node_embeddings.device

        dense = torch.zeros(B, max_len, D, device=device)
        offset = 0
        for i in range(B):
            nc = node_counts[i].item()
            dense[i, :nc] = node_embeddings[offset:offset + nc]
            offset += nc

        return dense

    def forward(self, adj, coords, node_features=None, **kwargs):
        """
        Args:
            adj: (B, max_len, max_len) dense adjacency matrix
            coords: (B, max_len, 3) normalized coordinates
            node_features: (B, max_len, F) optional node features (e.g. ESM embeddings)
            **kwargs: may contain edge_index, batch_vec, node_counts from sparse collate

        Returns:
            (B, max_len, out_dim) structure embeddings
        """
        B, max_len, _ = coords.shape

        # Get sparse representation
        edge_index_kw = kwargs.get('edge_index')
        batch_vec_kw = kwargs.get('batch_vec')
        node_counts_kw = kwargs.get('node_counts')

        if edge_index_kw is not None and batch_vec_kw is not None and node_counts_kw is not None:
            # Use pre-computed sparse data from collate
            edge_index = edge_index_kw
            batch_vec = batch_vec_kw
            node_counts = node_counts_kw

            # Flatten dense tensors to match sparse format
            # node_features and coords are (B, max_len, F) — extract real nodes
            all_pos = []
            all_x = []
            offset = 0
            for i in range(B):
                nc = node_counts[i].item()
                all_pos.append(coords[i, :nc])
                if node_features is not None:
                    all_x.append(node_features[i, :nc])

            pos = torch.cat(all_pos, dim=0)
            x = torch.cat(all_x, dim=0) if node_features is not None else None
        else:
            # Fallback: convert dense to sparse
            pos, x, edge_index, batch_vec, node_counts = self._dense_to_sparse(
                adj, coords, node_features
            )

        # Use node_features (ESM embeddings) as input if available, else coords
        if x is not None:
            h_V = self.emb_in(x)
        else:
            h_V = self.emb_in(pos)

        # Initial edge features
        h_E = self._compute_edge_features(pos, edge_index)
        h_E = self.W_e_d[0](h_E)

        # Initial node features: scalar only -> (scalar, vector)
        h_V = self.W_v(h_V)

        graphs = batch_vec

        # ==================== Encoder (downsampling) path ====================
        stack_down_idx = []
        stack_down_h_V = []
        stack_down_edges = []
        stack_down_batch = []
        stack_down_pos = []

        idx = torch.arange(pos.size(0), dtype=torch.long, device=pos.device)
        stack_down_idx.append(idx)

        for i, layer in enumerate(self.layers_d):
            if i % 2 == 0:
                # Save state for skip connection
                stack_down_h_V.append(h_V)
                stack_down_batch.append(graphs)
                stack_down_pos.append(pos)
                stack_down_edges.append(edge_index)

                # FPS pooling
                idx = fps(pos, graphs, 0.6)
                stack_down_idx.append(idx)

                # Pool node features
                h_s = self.reds[i // 2](h_V[0], edge_index)[idx]
                row, col = edge_index
                h_v = scatter_add(
                    h_V[1][row], col, dim=0,
                    dim_size=h_V[1].size(0),
                )[idx]
                h_V = (h_s, h_v)
                pos = pos[idx]
                graphs = graphs[idx]

                # Recompute edges on pooled graph
                edge_index = knn_graph(pos, k=self.k, batch=graphs)

                # Recompute edge features
                h_E = self._compute_edge_features(pos, edge_index)
                h_E = self.W_e_d[i // 2](h_E)

            h_V = layer(h_V, edge_index, h_E)

        # ==================== Decoder (upsampling) path ====================
        for i, layer in enumerate(self.layers_up):
            if i % 2 == 0:
                # Restore skip connection state
                h_V_skip = stack_down_h_V.pop()
                graphs = stack_down_batch.pop()
                pos = stack_down_pos.pop()
                idx = stack_down_idx.pop()
                edge_index = stack_down_edges.pop()

                # Recompute edge features at this resolution
                h_E = self._compute_edge_features(pos, edge_index)
                h_E = self.W_e_up[i // 2](h_E)

                h_s, h_v = h_V
                h_s_skip, h_v_skip = h_V_skip

                # Upsample + skip connection
                mask_s = torch.ones(h_s_skip.shape[0], dtype=torch.bool, device=h_s.device)
                mask_s[idx] = False

                mask_v = torch.ones(h_v_skip.shape[0], dtype=torch.bool, device=h_v.device)
                mask_v[idx] = False

                # Concatenate upsampled features with skip features
                h_new_s = torch.zeros(
                    h_s_skip.shape[0], h_s_skip.shape[1] + h_s.shape[1],
                    device=h_s.device
                )
                h_new_v = torch.zeros(
                    h_v_skip.shape[0], h_v_skip.shape[1] + h_v.shape[1], h_v_skip.shape[2],
                    device=h_v.device
                )

                # Nodes that were sampled: concatenate pooled + skip
                h_new_s[idx] = torch.cat((h_s, h_s_skip[idx]), dim=1)
                h_new_v[idx] = torch.cat((h_v, h_v_skip[idx]), dim=1)

                # Nodes that were NOT sampled: use skip only (zero + skip)
                h_new_s[mask_s, h_s.shape[1]:] = h_s_skip[mask_s]
                h_new_v[mask_v, h_v.shape[1]:] = h_v_skip[mask_v]

                # Project concatenated features back to original dim
                h_s = self.lin_s[i // 2](h_new_s)
                h_v = self.lin_v[i // 2](h_new_v.transpose(-1, -2)).transpose(-1, -2)
                h_V = (h_s, h_v)

            h_V = layer(h_V, edge_index, h_E)

        # Output projection: (total_nodes, s_dim) scalar features
        out = self.W_out(h_V)  # (total_nodes, s_dim)

        # Project to target out_dim
        out = self.final_proj(out)  # (total_nodes, out_dim)

        # Convert back to dense format
        dense_out = self._sparse_to_dense(out, batch_vec, node_counts, max_len)

        return dense_out  # (B, max_len, out_dim)
