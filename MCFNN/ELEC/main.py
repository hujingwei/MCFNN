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
import copy

from model import MCFNN,  mutual_information

warnings.filterwarnings('ignore', category=UserWarning)


class MoleculeDataset(Dataset):
    """
    Dataset class for molecular data
    """

    def __init__(self, dgl_graphs, proportions, labels, num_atom, SMILES_Embeding, graph_edge_list, temperatures):
        self.graphs = dgl_graphs
        self.proportions = proportions
        self.labels = labels
        self.num_atom = num_atom
        self.emb = SMILES_Embeding
        self.graph_edge_list = graph_edge_list
        self.temperatures = temperatures

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (self.graphs[idx], torch.tensor(self.proportions[idx]), torch.tensor(self.labels[idx]),
                torch.tensor(self.num_atom[idx]), self.emb[idx], self.graph_edge_list[idx],
                torch.tensor(self.temperatures[idx]))


def regress(model, gs, proportions, num_atoms, emb, edge, temperatures, device):
    """
    Regression function for model inference

    Args:
        model: CombinedModel instance
        gs: DGL graphs
        proportions: Proportion data
        num_atoms: Number of atoms
        emb: SMILES embeddings
        edge: Edge data
        temperatures: Temperature data
        device: Computation device

    Returns:
        Model outputs
    """
    gs = gs.to(device)
    proportions = proportions.to(device)
    num_atoms = num_atoms.to(device)
    emb = emb.to(device)
    temperatures = temperatures.to(device)
    edge = [e.to(device) for e in edge] if isinstance(edge[0], torch.Tensor) else edge

    h = gs.ndata.pop('n_feat').to(device)
    e = gs.edata.pop('e_feat').to(device)
    d = gs.edata.pop('distance').to(device)

    e1 = torch.concat([e, d], dim=1).to(device)
    return model(gs, h, e1, proportions, num_atoms, emb, edge, temperatures)


def setup_seed(seed):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    np.random.seed(seed)
    random.seed(seed)


def load_data():
    """
    Load and preprocess all data
    """
    print("Loading data...")

    # Load data files
    graph_edge_list = torch.load('Liquid_graph_edge_list.pt')
    graph_list = torch.load('Liquid_graph_list.pt')
    label_data = torch.load('Liquid_Electropy_labels_List.pt')
    proportions = torch.load('Liquid_Electropy_Comp_List.pt')
    SMILES_Embeding = torch.load('Liquid_Electropy_Text_Embeding_Tensor.pt')
    temperatures = torch.load('Liquid_Electropy_Temperature.pt')
    mol_graph = torch.load('Liquid_Electropy_DGL.pt')

    # Process labels
    labels_float = torch.stack([torch.tensor(label, dtype=torch.float32) for label in label_data])
    label = labels_float  # Use raw values

    # Process proportions
    max_length = 6
    proportions = [row + [0] * (max_length - len(row)) for row in proportions]

    # Process embeddings
    SMILES_Embeding = SMILES_Embeding.transpose(0, 1)

    # Count atoms
    num_atom = []
    for i in range(len(mol_graph)):
        num = len(mol_graph[i])
        num_atom.append(num)

    return (graph_list, proportions, label, num_atom, SMILES_Embeding,
            graph_edge_list, temperatures)


def main():
    """
    Main training function
    """
    # Device configuration
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    (graph_list, proportions, label, num_atom,
     SMILES_Embeding, graph_edge_list, temperatures) = load_data()

    # Set random seed
    seed = 108
    setup_seed(seed)
    print(f"Using seed: {seed}")

    # Split data
    (graph_train, graph_test, proportions_train, proportions_test,
     label_train, label_test, num_atom_train, num_atom_test,
     emb_train, emb_test, edge_train, edge_test,
     temp_train, temp_test) = train_test_split(
        graph_list, proportions, label, num_atom, SMILES_Embeding,
        graph_edge_list, temperatures, test_size=0.1, random_state=seed)

    # Create datasets
    train_dataset = MoleculeDataset(graph_train, proportions_train, label_train,
                                    num_atom_train, emb_train, edge_train, temp_train)
    test_dataset = MoleculeDataset(graph_test, proportions_test, label_test,
                                   num_atom_test, emb_test, edge_test, temp_test)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders
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
            [item[5] for item in batch],
            torch.stack([item[6] for item in batch])
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
            [item[5] for item in batch],
            torch.stack([item[6] for item in batch])
        )
    )

    # Initialize model
    model = MCFNN().to(device)
    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)

    # Training parameters
    train_best_mse = 999
    test_best_mse = 999
    train_best_r = -9999
    test_best_r = -9999
    best_model_weights = None
    best_epoch = 0

    print("Starting training...")

    # Training loop
    for epoch in range(1800):
        start_time = time.time()
        train_all_loss = []
        train_prediction = []
        train_labels = []

        model.train()
        for (batched_graph, proportions_1, labels_1, num_atoms,
             emb, train_edge, train_temp) in train_dataloader:
            labels_1 = torch.tensor(labels_1).unsqueeze(1).clone().detach()
            prediction_1, rob_prediction, cross_rob, loss_sim, _ = regress(
                model, batched_graph, proportions_1, num_atoms, emb,
                train_edge, train_temp, device)

            prediction_1 = prediction_1.cpu()
            loss_sim = loss_sim.cpu()

            [train_prediction.append(val[0].item()) for idx, val in enumerate(prediction_1[:])]
            [train_labels.append(val) for idx, val in enumerate(labels_1)]

            train_loss = (loss_fn(prediction_1, labels_1)).mean()
            train_all_loss.append(train_loss.item())

            train_loss_ = train_loss + loss_sim

            optimizer.zero_grad()
            train_loss_.backward()
            optimizer.step()

        # Calculate training metrics
        train_aver_error = sum(train_all_loss) / len(train_all_loss)
        train_predictions_flattened = torch.tensor(train_prediction).view(-1)
        train_labels_tensor = torch.tensor(train_labels).view(-1)
        train_mse = torch.mean((train_predictions_flattened - train_labels_tensor) ** 2)
        train_rmse = torch.sqrt(train_mse)
        train_r_squared = r2_score(train_labels_tensor, train_predictions_flattened.numpy())

        # Save best model
        if train_mse < train_best_mse:
            best_epoch = epoch
            train_best_mse = train_mse
            train_best_r = train_r_squared
            train_best_rmse = train_rmse
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