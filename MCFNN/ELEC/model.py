import copy
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GATv2Conv


import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def get_gnn_layer(name, in_channels, out_channels, heads):
    """
    Get GNN layer based on the specified name
    """
    if name == 'gcn':
        layer = GCNConv(in_channels, out_channels)
    elif name == 'gat':
        layer = GATConv(-1, out_channels, heads)
    elif name == 'sage':
        layer = SAGEConv(in_channels, out_channels)
    elif name == 'gin':
        layer = GINConv(Linear(in_channels, out_channels), train_eps=True)
    elif name == 'gat2':
        layer = GATv2Conv(-1, out_channels, heads)
    else:
        raise ValueError(f"Unsupported GNN layer: {name}")
    return layer


class MCFNN(nn.Module):
    """
    Combined Model for molecular property prediction using graph neural networks
    """

    def __init__(self):
        super(MCFNN, self).__init__()

        self.dropout = nn.Dropout(0.2)

        self.emb_Linear = nn.Linear(384, 64)
        self.emb_Linear_2 = nn.Linear(385, 512)
        self.fc_combined1 = nn.Linear(193, 256)  # 32+64
        self.fc_combined2 = nn.Linear(256, 1)

        # GNN layers for molecular graphs
        lay = 'gcn'
        self.convs = nn.ModuleList()
        for i in range(2):
            first_channels = 32 if i == 0 else 32
            second_channels = 64 if i == 2 - 1 else 32
            heads = 1 if i == 2 - 1 or 'gat' not in lay else 8
            self.convs.append(get_gnn_layer(lay, first_channels, second_channels, heads))

        # GNN layers for molecular aggregation
        self.convs_mol = nn.ModuleList()
        for i in range(2):
            first_channels = 64 if i == 0 else 128
            second_channels = 64 if i == 2 - 1 else 128
            heads = 1 if i == 2 - 1 or 'gat' not in lay else 8
            self.convs_mol.append(get_gnn_layer(lay, first_channels, second_channels, heads))

        # GNN layers for formula-level processing
        self.convs_formular = nn.ModuleList()
        for i in range(2):
            first_channels = 64 if i == 0 else 128
            second_channels = 64 if i == 2 - 1 else 128
            heads = 1 if i == 2 - 1 or 'gat' not in lay else 8
            self.convs_formular.append(get_gnn_layer(lay, first_channels, second_channels, heads))

        # GNN layers for text embeddings
        self.convs_text = nn.ModuleList()
        for i in range(2):
            first_channels = 384 if i == 0 else 128
            second_channels = 64 if i == 2 - 1 else 128
            heads = 1 if i == 2 - 1 or 'gat' not in lay else 8
            self.convs_text.append(get_gnn_layer(lay, first_channels, second_channels, heads))

        # Transformer encoders for attention mechanisms
        self.formular_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dropout=0.8), num_layers=6
        )
        self.text_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dropout=0.8), num_layers=6
        )
        self.pool_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=385, nhead=7, dropout=0.8), num_layers=6
        )
        self.pool_attention_sim = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dropout=0.8), num_layers=6
        )

        # Pooling MLP layers
        self.pool_MLP_1 = nn.Linear(385, 128)
        self.pool_MLP_2 = nn.Linear(128, 64)

    def forward(self, g, node_feats_0, edge_feats, x, num_atoms, emb, edge_list, temperatures):
        """
        Forward pass of the combined model

        Args:
            g: DGL graph
            node_feats_0: Initial node features
            edge_feats: Edge features
            x: Proportion data
            num_atoms: Number of atoms per molecule
            emb: SMILES embeddings
            edge_list: Edge list for molecular graphs
            temperatures: Temperature data

        Returns:
            combined_out: Model predictions
            placeholder1: Placeholder for compatibility
            placeholder2: Placeholder for compatibility
            similarity_loss: Similarity regularization loss
            placeholder3: Placeholder for compatibility
        """
        # Process molecular graphs
        sum_before = 0
        sum_after = 0
        x[torch.isnan(x)] = 0.0
        mols_out = []
        node_feats = torch.clone(node_feats_0)

        for idx, edgs in enumerate(edge_list):
            for jdx, edge in enumerate(edgs):
                length = len(edge[0])
                sum_after = sum_after + length
                virtual = torch.ones(1, length)
                try:
                    edge_virtual = torch.cat([torch.tensor(edge), virtual], dim=0)
                    edge_virtual = torch.cat([edge_virtual, torch.ones(length + 1, 1)], dim=1)
                    edge_virtual[length, length] = 0
                    edges = torch.nonzero(edge_virtual, as_tuple=False).T
                    node_feat = node_feats[sum_before:sum_after]
                    node_feat = torch.cat([node_feat, torch.zeros(1, len(node_feat[0]), device=node_feat.device)],
                                          dim=0)
                except Exception as e:
                    print(f"Error processing edge: {e}")
                    continue

                mol_out = node_feat
                for i, conv in enumerate(self.convs):
                    mol_out = F.relu(conv(mol_out, edges.to(node_feat.device)))

                if idx == 0 and jdx == 0:
                    mols_out = mol_out[length]
                    mols_out = mols_out.unsqueeze(0)
                else:
                    mols_out = torch.cat([mols_out, mol_out[length].unsqueeze(0)], dim=0)

                sum_before = sum_before + length

        # Create indices for aggregation
        indices = torch.zeros(mols_out.size(0), dtype=torch.long)
        start_index = 0
        for i, size in enumerate(num_atoms):
            end_index = start_index + size
            indices[start_index:end_index] = i
            start_index = end_index

        # Process embeddings and calculate similarity
        emb_0 = torch.clone(emb)
        emb_0 = emb_0.view(-1, 384)
        mask = emb_0 != 0 * 384
        non_zero_rows = mask.any(dim=1)
        emb_1 = torch.clone(emb_0[non_zero_rows])
        sim_emb_1 = self.emb_Linear(emb_1)
        sim_Mol = mutual_information(mols_out, sim_emb_1)

        # Pooling mechanism with concentration information
        emb_pre_pool = torch.clone(emb_0[non_zero_rows])
        emb_pool = torch.cat([emb_pre_pool.to(emb_pre_pool.device),
                              torch.zeros(emb_pre_pool.size(0), 1).to(emb_pre_pool.device)], dim=1)

        mp_s = 0
        for i, concs in enumerate(x):
            length = 0
            for j in indices:
                if j == i:
                    length = length + 1
            for k, conc in enumerate(concs):
                if conc != 0 and k < length:
                    emb_pool[mp_s][emb_pool.size(1) - 1] = torch.tensor(conc)
                    mp_s = mp_s + 1
                else:
                    break

        # Formula-level pooling
        pool_start = 0
        pool_end = 0
        formular_pools = []
        for idx, mols_num in enumerate(num_atoms):
            pool_end = pool_end + mols_num
            aggregate_pool = self.pool_attention(emb_pool[pool_start:pool_end])
            aggregate_pool = self.pool_MLP_1(aggregate_pool)
            formular_pool = torch.sum(aggregate_pool, dim=0)
            formular_pool = self.pool_MLP_2(formular_pool)

            if idx == 0:
                formular_pools = formular_pool.unsqueeze(0)
            else:
                formular_pools = torch.cat([formular_pools, formular_pool.unsqueeze(0)], dim=0)
            pool_start = pool_start + mols_num

        # Molecular aggregation with concentration weights
        mol_agg_before = 0
        mol_agg_after = 0
        formula_mol_features = []
        text_mol_features = []
        rob_formula_mol_features = []

        for idx, formula_mol_num in enumerate(num_atoms):
            mol_agg_after = mol_agg_after + formula_mol_num
            mol_conc = x[idx, 0:formula_mol_num]
            degree = torch.tensor(1 / (formula_mol_num - 1)).to(mol_conc.device)

            # Create concentration-weighted edges
            mol_conc_edge = torch.ones(formula_mol_num, formula_mol_num).to(mol_conc.device)
            mol_conc_edge = degree * mol_conc_edge + (1 - degree) * torch.eye(formula_mol_num).to(mol_conc.device)
            mol_conc_edge = torch.cat([mol_conc_edge, torch.ones(1, formula_mol_num).to(mol_conc.device)], dim=0)
            for jdx, conc_mol in enumerate(mol_conc):
                mol_conc_edge[formula_mol_num, jdx] = conc_mol
            mol_conc_edge = torch.cat([mol_conc_edge, torch.zeros(formula_mol_num + 1, 1).to(mol_conc.device)], dim=1)
            for jdx, conc_mol in enumerate(mol_conc):
                mol_conc_edge[jdx, formula_mol_num] = conc_mol

            # Process formula molecular features
            formula_mol_feature = mols_out[mol_agg_before:mol_agg_after]
            formula_mol_feature = torch.cat([formula_mol_feature,
                                             torch.zeros(1, mols_out.size(1)).to(formula_mol_feature.device)], dim=0)

            # Process text molecular features
            text_mol_feature = torch.clone(emb[idx][0:formula_mol_num])
            text_mol_feature = torch.cat([text_mol_feature,
                                          torch.zeros(1, emb[idx][0:formula_mol_num].size(1)).to(
                                              text_mol_feature.device)],
                                         dim=0)

            formula_mol_agg_edge_sparse = torch.nonzero(mol_conc_edge, as_tuple=False).T
            rob_formula_mol_feature = []

            # Apply GNN layers
            for i, conv in enumerate(self.convs_mol):
                rob_formula_mol_feature = torch.clone(formula_mol_feature)
                formula_mol_feature = F.relu(conv(formula_mol_feature, formula_mol_agg_edge_sparse))

                # Robustness: apply perturbed version
                rob_conv = copy.deepcopy(conv)
                params = rob_conv.state_dict()
                perturbed_params = {key: value + torch.randn_like(value) * 1.0 for key, value in params.items()}
                rob_conv.load_state_dict(perturbed_params)
                rob_formula_mol_feature = F.relu(rob_conv(rob_formula_mol_feature, formula_mol_agg_edge_sparse))

            # Process text features
            for i, conv in enumerate(self.convs_text):
                text_mol_feature = F.relu(conv(text_mol_feature, formula_mol_agg_edge_sparse))

            # Aggregate results
            if idx == 0:
                formula_mol_features = formula_mol_feature[formula_mol_num]
                formula_mol_features = formula_mol_features.unsqueeze(0)
                rob_formula_mol_features = rob_formula_mol_feature[formula_mol_num]
                rob_formula_mol_features = rob_formula_mol_features.unsqueeze(0)
                text_mol_features = text_mol_feature[formula_mol_num]
                text_mol_features = text_mol_features.unsqueeze(0)
            else:
                formula_mol_features = torch.cat([formula_mol_features,
                                                  formula_mol_feature[formula_mol_num].unsqueeze(0)], dim=0)
                rob_formula_mol_features = torch.cat([rob_formula_mol_features,
                                                      rob_formula_mol_feature[formula_mol_num].unsqueeze(0)], dim=0)
                text_mol_features = torch.cat([text_mol_features,
                                               text_mol_feature[formula_mol_num].unsqueeze(0)], dim=0)

            mol_agg_before = mol_agg_after

        result_2 = formula_mol_features

        # Calculate similarity losses using attention mechanisms
        clone_graph_out = torch.clone(result_2)
        clone_text_mol_features = torch.clone(text_mol_features)
        clone_text_pool_features = torch.clone(formular_pools)

        sim_graph_out = self.formular_attention(clone_graph_out)
        sim_text_mol_features = self.text_attention(clone_text_mol_features)
        sim_text_pool_features = self.pool_attention_sim(clone_text_pool_features)

        # Calculate mutual information for regularization
        multual_formular_1 = mutual_information(sim_graph_out, sim_text_mol_features)
        multual_formular_2 = mutual_information(sim_graph_out, sim_text_pool_features)
        multual_formular_3 = mutual_information(sim_text_mol_features, sim_text_pool_features)

        # Combine features for final prediction
        combined_out = torch.cat((result_2, formular_pools), dim=-1)
        combined_out = torch.cat((combined_out, text_mol_features), dim=-1)
        temperatures_1 = torch.clone(temperatures)
        combined_out = torch.cat([combined_out, temperatures_1.unsqueeze(1)], dim=-1)

        # Final prediction layers
        combined_out = F.relu(self.fc_combined1(combined_out))
        combined_out = self.fc_combined2(combined_out)

        return combined_out, 0., 0., (sim_Mol - (multual_formular_1 + multual_formular_2 + multual_formular_3)), 0.


def NearestNeighbors(x: torch.Tensor, k):
    """
    Calculate nearest neighbors distances for mutual information estimation
    """
    neighbors_distances = []
    for i in range(x.size(0)):
        distances = []
        n = 0
        for j in range(x.size(0)):
            if i != j and n == 0:
                distances = torch.max(torch.abs(x[i, :] - x[j, :])).unsqueeze(0).to(x.device)
                n = 1
            elif i != j and n != 0:
                distances = torch.cat([distances, torch.max(torch.abs(x[i, :] - x[j, :])).unsqueeze(0)], dim=0).to(
                    x.device)
                n = 1
            elif i == j and n != 0 and i != 0:
                distances = torch.cat([distances.to(x.device), torch.tensor(0).unsqueeze(0).to(x.device)], dim=0).to(
                    x.device)
                n = 1
            elif i == j and n == 0 and i == 0:
                distances = torch.tensor(0).unsqueeze(0).to(x.device)
                n = 1
            else:
                distances = torch.cat([distances, torch.tensor(0).unsqueeze(0)], dim=0).to(x.device)
                n = 1
        sorted_distances, indices = torch.topk(distances.view(-1), k + 1)
        if i == 0:
            neighbors_distances = sorted_distances.unsqueeze(0).to(x.device)
        else:
            neighbors_distances = torch.cat([neighbors_distances, sorted_distances.unsqueeze(0)], dim=0)
    return neighbors_distances


def mutual_information(x: torch.Tensor, y: torch.Tensor, k: int = 3) -> float:
    """
    Calculate the mutual information between two tensors using the k-nearest neighbor approach.

    Parameters:
    x (torch.Tensor): Tensor of shape (n_samples, n_features_x).
    y (torch.Tensor): Tensor of shape (n_samples, n_features_y).
    k (int): Number of nearest neighbors to use in the MI estimation. Default is 3.

    Returns:
    float: Estimated mutual information between x and y.
    """
    n_samples = x.size(0)

    # Combine x and y to calculate distances in the joint space
    xy = torch.cat([x, y], dim=1)

    # Fit nearest neighbors in the joint space
    nn_xy = NearestNeighbors(xy, k)
    nn_x = NearestNeighbors(x, k)
    nn_y = NearestNeighbors(y, k)

    # Find the k-th nearest neighbor distance in the joint space
    distances_xy = nn_xy
    distances_x = nn_x
    distances_y = nn_y

    # Get the k-th nearest distance
    k_distance_xy = distances_xy[:, 0]
    k_distance_x = distances_x[:, 0]
    k_distance_y = distances_y[:, 0]

    n_x = torch.sum(distances_x <= k_distance_xy.unsqueeze(1).repeat(1, distances_x.size(1)), dim=1) - 1
    n_y = torch.sum(distances_y <= k_distance_xy.unsqueeze(1).repeat(1, distances_x.size(1)), dim=1) - 1

    # Calculate the mutual information
    mi_estimate = (torch.log(torch.tensor(n_samples)) + torch.log(torch.tensor(k)) -
                   torch.mean(torch.log(n_x + 1) + torch.log(n_y + 1)))

    return mi_estimate




