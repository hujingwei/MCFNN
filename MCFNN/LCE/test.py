import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from main import MoleculeDataset
from main import regress
from main import setup_seed
from main import preprocess_data
import numpy as np
import random
import warnings
import os

from model import MCFNN

warnings.filterwarnings('ignore', category=UserWarning)


def test():
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


    seed = 109
    setup_seed(seed)

    # Data Segmentation
    _, graph_test, _, proportions_test, _, label_test, _, num_atom_test, _, emb_test, _, edge_test = train_test_split(
        graph_list, proportions, label, num_atom, SMILES_Embeding, graph_edge_list, test_size=0.1, random_state=seed)

    # Create test dataset
    test_dataset = MoleculeDataset(graph_test, proportions_test, label_test, num_atom_test, emb_test, edge_test)

    # Create test DataLoader
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

    # Load trained model
    checkpoint_path = f'best_model_weights_{seed}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully. Training epoch: {checkpoint['epoch']}")
        print(f"Training loss: {checkpoint['train_loss']:.6f}, R²: {checkpoint['train_r2']:.6f}")
    else:
        print(f"Error: Model file not found {checkpoint_path}")
        return

    # Testing
    model.eval()
    test_labels = []
    test_prediction = []

    with torch.no_grad():
        for batched_graph, proportions_2, labels, num_atoms, emb, train_edge in test_dataloader:
            labels = labels.tolist()
            prediction, _, _, _, _ = regress(model, batched_graph, proportions_2, num_atoms, emb, train_edge, device)
            prediction = prediction.cpu()
            prediction = prediction.squeeze(1).tolist()

            for i in range(len(labels)):
                test_labels.append(labels[i])
                test_prediction.append(prediction[i])

    # Calculate test metrics
    predictions_flattened = torch.tensor(test_prediction).view(-1)
    test_labels_tensor = torch.tensor(test_labels).view(-1)
    mse = torch.mean((predictions_flattened - test_labels_tensor) ** 2)
    rmse = torch.sqrt(mse)
    r_squared = r2_score(test_labels_tensor.numpy(), predictions_flattened.numpy())


    print('Test results:')
    print(f'MSE: {mse.item():.6f}')
    print(f'RMSE: {rmse.item():.6f}')
    print(f'R²: {r_squared:.6f}')


    # Save results
    torch.save(test_labels_tensor.numpy(), f'test_labels_{seed}.pt')
    torch.save(predictions_flattened.numpy(), f'test_predictions_{seed}.pt')
    print(f"Test results have been saved.: test_labels_{seed}.pt, test_predictions_{seed}.pt")


if __name__ == '__main__':
    test()