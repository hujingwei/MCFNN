import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from main import MoleculeDataset
from main import regress
from main import setup_seed
from main import load_data
import numpy as np
import random
import warnings
import os
import matplotlib.pyplot as plt

from model import MCFNN

warnings.filterwarnings('ignore', category=UserWarning)


def test():
    """
    Main testing function
    """
    # Device configuration
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    (graph_list, proportions, label, num_atom,
     SMILES_Embeding, graph_edge_list, temperatures) = load_data()

    # Set random seed (same as training)
    seed = 108
    setup_seed(seed)
    print(f"Using seed: {seed}")

    # Split data (same as training)
    (_, graph_test, _, proportions_test, _, label_test,
     _, num_atom_test, _, emb_test, _, edge_test,
     _, temp_test) = train_test_split(
        graph_list, proportions, label, num_atom, SMILES_Embeding,
        graph_edge_list, temperatures, test_size=0.1, random_state=seed)

    # Create test dataset
    test_dataset = MoleculeDataset(graph_test, proportions_test, label_test,
                                   num_atom_test, emb_test, edge_test, temp_test)

    print(f"Test samples: {len(test_dataset)}")

    # Create test data loader
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

    # Load trained model
    checkpoint_path = f'best_model_liquid_seed_{seed}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully from {checkpoint_path}")
        print(f"Training epoch: {checkpoint['epoch']}")
        print(f"Training loss: {checkpoint['train_loss']:.8f}")
        print(f"Training R²: {checkpoint['train_r2']:.8f}")
    else:
        print(f"Error: Model file not found at {checkpoint_path}")
        return

    # Testing
    model.eval()
    test_labels = []
    test_predictions = []

    print("Starting testing...")

    with torch.no_grad():
        for (batched_graph, proportions_2, labels, num_atoms,
             emb, train_edge, test_temp) in test_dataloader:

            labels_list = labels.tolist()
            prediction, _, _, _, _ = regress(
                model, batched_graph, proportions_2, num_atoms,
                emb, train_edge, test_temp, device)

            prediction = prediction.cpu()
            prediction_list = prediction.squeeze(1).tolist()

            for i in range(len(labels_list)):
                test_labels.append(labels_list[i])
                test_predictions.append(prediction_list[i])

    # Calculate test metrics
    predictions_tensor = torch.tensor(test_predictions).view(-1)
    labels_tensor = torch.tensor(test_labels).view(-1)

    mse = torch.mean((predictions_tensor - labels_tensor) ** 2)
    rmse = torch.sqrt(mse)
    r_squared = r2_score(labels_tensor.numpy(), predictions_tensor.numpy())

    # Print results
    print('=' * 60)
    print('TEST RESULTS:')
    print('=' * 60)
    print(f'MSE:  {mse.item():.8f}')
    print(f'RMSE: {rmse.item():.8f}')
    print(f'R²:   {r_squared:.8f}')
    print('=' * 60)

    # Plot results
    plot_predictions(labels_tensor.numpy(), predictions_tensor.numpy(), seed, "test")

    # Save results
    torch.save(labels_tensor.numpy(), f'test_labels_{seed}.pt')
    torch.save(predictions_tensor.numpy(), f'test_predictions_{seed}.pt')
    print(f"Results saved: test_labels_{seed}.pt, test_predictions_{seed}.pt")


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
    label = labels_float

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


def plot_predictions(true_labels, predictions, seed, dataset_type="test"):
    """
    Plot true values vs predictions
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(true_labels, predictions, alpha=0.6)
    plt.plot([min(true_labels), max(true_labels)],
             [min(true_labels), max(true_labels)],
             'r--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{dataset_type.capitalize()} Set Predictions (Seed {seed})')
    plt.legend()
    plt.savefig(f'true_vs_pred_{dataset_type}_seed_{seed}.png')
    plt.close()
    print(f"Prediction plot saved: true_vs_pred_{dataset_type}_seed_{seed}.png")


if __name__ == '__main__':
    test()