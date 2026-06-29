import argparse
import pandas
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from typing import Union

from globals import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    TRAINING_DATA_CSV,
)
from helpers import get_device
from models import MODELS


# Helpers #

def load_training_data(path: Union[str, Path] = TRAINING_DATA_CSV) -> pandas.DataFrame:
    dataframe = pandas.read_csv(path)
    if "Unnamed: 0" in dataframe.columns:
        dataframe = dataframe.drop(columns=["Unnamed: 0"])
    return dataframe

def resolve_model(model_name: str) -> nn.Module:
    """
    Resolve the model class from the model name.
    """
    if model_name in MODELS:
        return MODELS[model_name]
    else:
        for model_class in MODELS.values():
            if model_class.__name__ == model_name:
                return model_class

        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Available models: {', '.join(sorted(MODELS))}"
        )

def save_artifacts(
    model: nn.Module,
    model_path: Union[str, Path],
    means: torch.Tensor,
    stds: torch.Tensor,
    normalization_path: Union[str, Path],
) -> None:
    """
    Saves the model and normalization stats to the specified paths.

    Args:
        model: torch.nn.Module
            The model to save.
        model_path: Union[str, Path]
            The path to save the model to.
        means: torch.Tensor
            The means of the features.
        stds: torch.Tensor
            The standard deviations of the features.
        normalization_path: Union[str, Path]
            The path to save the normalization stats to.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    torch.save(
        {"means": means, "stds": stds},
        normalization_path,
    )
    tqdm.write(f"Saved model to {model_path}")
    tqdm.write(f"Saved normalization stats to {normalization_path}")


# Training #

def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Evaluates the performance of a model on a data loader.

    Args:
        model: torch.nn.Module
            The model to evaluate.
        device: torch.device
            The device to use for evaluation.
        data_loader: torch.utils.data.DataLoader
            The data loader to use for evaluation.
        criterion: torch.nn.Module
            The criterion to use for evaluation.

    Returns:
        tuple[float, float]
            The total loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            total_loss += criterion(logits, batch_labels).item() * batch_labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    return total_loss / total, (correct / total) * 100

def train(
    training_data: pandas.DataFrame,
    model: nn.Module,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
) -> None:
    """
    Train the model using the training data and cross-entropy loss.

    Args:
        training_data: pandas.DataFrame
            The training data to use.
        model_class: nn.Module
            The class of the model to train.
        num_epochs: int
            The number of epochs to train for.
        batch_size: int
            The batch size to use.
        learning_rate: float
            The learning rate to use.
        val_fraction: float
            The fraction of the training data to use for validation.
    """
    device = get_device()
    tqdm.write(f"Using device: {device}")

    features = torch.tensor(
        training_data[FEATURE_COLUMNS].values,
        dtype=torch.float32,
    )
    labels = torch.tensor(
        training_data[LABEL_COLUMNS].values.argmax(axis=1),
        dtype=torch.long,
    )

    means = features.mean(dim=0)
    stds = features.std(dim=0)
    stds[stds == 0] = 1.0
    features = (features - means) / stds

    dataset = TensorDataset(features, labels)
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    epoch_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for _ in epoch_bar:
        model.train()
        train_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy = evaluate(model, device, val_loader, criterion)
        epoch_bar.set_postfix(
            train_loss=f"{train_loss / len(train_loader):.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_accuracy:.2f}%",
        )

    tqdm.write(f"Finished training in {round(time.time() - start_time, 1)} seconds")
    model_path = Path("saved") / f"{model.__class__.__name__}.pt"
    normalization_path = Path("saved") / f"{model.__class__.__name__}_normalization.pt"
    save_artifacts(model, model_path, means, stds, normalization_path)


# Main #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fight outcome model.")
    parser.add_argument(
        "--model",
        default="linear",
        choices=sorted(MODELS),
        help="model architecture to train (default: linear)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="training batch size (default: 256)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="optimizer learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="fraction of data held out for validation (default: 0.1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    training_data = load_training_data()
    train(
        training_data,
        resolve_model(args.model),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
    )
