import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import random
import time
import warnings
import os

from model import MCFNN

warnings.filterwarnings('ignore', category=UserWarning)


class MoleculeDataset(Dataset):
    def __init__(self, dgl_graphs, proportions, labels, num_atom, SMILES_Embeding, graph_edge_list):
        self.graphs = dgl_graphs
        self.proportions = proportions
        self.labels = labels
        self.num_atom = num_atom
        self.emb = SMILES_Embeding
        self.graph_edge_list = graph_edge_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (self.graphs[idx], torch.tensor(self.proportions[idx]), torch.tensor(self.labels[idx]),
                torch.tensor(self.num_atom[idx]), self.emb[idx], self.graph_edge_list[idx])


def regress(model, gs, proportions, num_atoms, emb, edge, device):
    gs = gs.to(device)
    proportions = proportions.to(device)
    num_atoms = num_atoms.to(device)
    emb = emb.to(device)
    edge = [e.to(device) for e in edge] if isinstance(edge[0], torch.Tensor) else edge

    h = gs.ndata.pop('n_feat').to(device)
    e = gs.edata.pop('e_feat').to(device)
    d = gs.edata.pop('distance').to(device)

    e1 = torch.concat([e, d], dim=1).to(device)
    return model(gs, h, e1, proportions, num_atoms, emb, edge)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    np.random.seed(seed)
    random.seed(seed)


def preprocess_data(mol_graph):
    """Data Preprocessing Function"""
    max_length = 6
    proportions = torch.load('./data/LCE_All_conc_List.pt')
    proportions = [row + [0] * (max_length - len(row)) for row in proportions]

    graph_list = []
    num_atom = []
    graph_edge_list = []

    for i in range(len(mol_graph)):
        num = len(mol_graph[i])
        G_dgl = dgl.batch(mol_graph[i])
        mol_Gs = mol_graph[i]
        mols_edge = []

        for mol in mol_Gs:
            mol_edge = mol.adjacency_matrix().to_dense()
            mol_edata = mol.edata.pop('e_feat')
            mol_ndata = mol.ndata.pop('n_feat')
            node_num = mol.num_nodes()
            mol_edge = torch.zeros((node_num, node_num))

            for i_node in range(node_num):
                n = 999
                flag = True
                for j in range(node_num - 1):
                    j_num = j + (i_node) * (node_num - 1)
                    if j + (i_node) * (node_num - 1) < node_num * (node_num - 1):
                        now = mol_edata[j + (i_node) * (node_num - 1)]
                        if mol_edata[j + (i_node) * (node_num - 1)][4] == 0:
                            if j >= i_node:
                                if flag:
                                    n = j + 1
                                    idx_list = [idx + 1 for idx, value in
                                                enumerate(mol_edata[j + (i_node) * (node_num - 1)][0:3]) if value == 1]
                                    mol_edge[i_node, n] = idx_list[0] if len(idx_list) != 0 else 0
                                    n = n + 1
                                    flag = False
                                else:
                                    idx_list = [idx + 1 for idx, value in
                                                enumerate(mol_edata[j + (i_node) * (node_num - 1)][0:3]) if value == 1]
                                    mol_edge[i_node, n] = idx_list[0] if len(idx_list) != 0 else 0
                                    n = n + 1
                            else:
                                idx_list = [idx + 1 for idx, value in
                                            enumerate(mol_edata[j + (i_node) * (node_num - 1)][0:3]) if
                                            value == 1]
                                mol_edge[i_node, j] = idx_list[0] if len(idx_list) != 0 else 0
            mol_edge = mol_edge + mol_edge.t()
            mol_edge = mol_edge + torch.eye(mol_edge.size(0))
            mols_edge.append(torch.tensor(mol_edge))

        graph_edge_list.append(mols_edge)
        num_atom.append(num)
        graph_list.append(G_dgl)

    return graph_list, proportions, num_atom, graph_edge_list


def main():
    # Device Settings
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    mol_graph = torch.load('./data/LCE_dgl_Graph_List.pt')
    label = torch.load('./data/LCE_All_label.pt')
    SMILES_Embeding = torch.load('./data/LCE_Text_Embeding_Tensor.pt')
    SMILES_Embeding = SMILES_Embeding.transpose(0, 1)

    # Data Preprocessing
    graph_list, proportions, num_atom, graph_edge_list = preprocess_data(mol_graph)

    # Training Cycle
    seed = 109
    setup_seed(seed)

    # Data Segmentation
    graph_train, graph_test, proportions_train, proportions_test, label_train, label_test, num_atom_train, num_atom_test, emb_train, emb_test, edge_train, edge_test = train_test_split(
        graph_list, proportions, label, num_atom, SMILES_Embeding, graph_edge_list, test_size=0.1, random_state=seed)

    # Create a dataset
    train_dataset = MoleculeDataset(graph_train, proportions_train, label_train, num_atom_train, emb_train, edge_train)
    test_dataset = MoleculeDataset(graph_test, proportions_test, label_test, num_atom_test, emb_test, edge_test)

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=15,
        shuffle=True,
        collate_fn=lambda batch: (
            dgl.batch([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            torch.stack([item[2] for item in batch]),
            torch.stack([item[3] for item in batch]),
            torch.stack([item[4] for item in batch]),
            [item[5] for item in batch]
        )
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=15,
        shuffle=False,
        collate_fn=lambda batch: (
            dgl.batch([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            torch.stack([item[2] for item in batch]),
            torch.stack([item[3] for item in batch]),
            torch.stack([item[4] for item in batch]),
            [item[5] for item in batch]
        )
    )

    # Model Initialization
    model = MCFNN().to(device)
    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)

    # Training Parameters
    train_best_mse = 999
    train_best_r = -9999
    best_model_weights = None
    best_epoch = 0

    # Training Cycle
    for epoch in range(1801):
        start_time = time.time()
        train_all_loss = []
        train_labels = []
        train_prediction = []

        model.train()
        for batched_graph, proportions_1, labels_1, num_atoms, emb, train_edge in train_dataloader:
            labels_1 = torch.tensor(labels_1).unsqueeze(1).clone().detach()
            batched_graph_1 = batched_graph.clone()

            prediction_1, _, _, loss_sim, _ = regress(model, batched_graph, proportions_1, num_atoms, emb, train_edge,
                                                      device)
            prediction_1 = prediction_1.cpu()
            loss_sim = loss_sim.cpu()

            [train_prediction.append(val[0].item()) for idx, val in enumerate(prediction_1[:])]
            [train_labels.append(val) for idx, val in enumerate(labels_1)]

            train_loss = (loss_fn(prediction_1, labels_1)).mean()
            train_loss_ = train_loss + loss_sim

            train_all_loss.append(train_loss.item())

            optimizer.zero_grad()
            train_loss_.backward()
            optimizer.step()

        # Calculate training metrics
        train_aver_error = sum(train_all_loss) / len(train_all_loss)
        train_predictions_flattened = torch.tensor(train_prediction).view(-1)
        train_labels_tensor = torch.tensor(train_labels).view(-1)
        train_r_squared = r2_score(train_labels_tensor, train_predictions_flattened.numpy())

        # Save the optimal model
        if train_aver_error < train_best_mse:
            best_epoch = epoch
            train_best_mse = train_aver_error
            train_best_r = train_r_squared
            best_model_weights = model.state_dict().copy()
            # Save the optimal model
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_best_mse,
                'train_r2': train_best_r,
            }, f'best_model_weights_{seed}.pth')

        # Output training information
        epoch_time = time.time() - start_time
        if epoch % 100 == 0:
            print(f'Epoch: {epoch} | time-consuming: {epoch_time:.2f}seconds')
            print('Epoch: {}; Train: {:.6f}||Best: {:.6f}; Best_Epoch: {:d}'
                  .format(epoch, train_aver_error, train_best_mse, best_epoch))
            print('R²: Train: {:.6f}||Best: {:.6f}; \n'.format(train_r_squared, train_best_r))

    print(f'Training complete! The best model is saved at： best_model_weights_{seed}.pth')
    print(f'Best epoch: {best_epoch}, Best MSE: {train_best_mse:.6f}, Best R²: {train_best_r:.6f}')


if __name__ == '__main__':
    main()