import argparse
from datetime import date
from pathlib import Path
import torch
from typing import Optional

from data_handler import Data
from helpers import resolve_model
from models import MODELS
from globals import (
    INPUT_SIZE,
    LABEL_COLUMNS,
    MIN_FIGHTS,
    NUM_CLASSES,
)


class FightPredictor:
    def __init__(self, model: torch.nn.Module) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model().to(self.device)
        self.model_path = Path("saved") / f"{model.__class__.__name__}.pt"
        self.normalization_path = (
            Path("saved") / f"{model.__class__.__name__}_normalization.pt"
        )
        self.means = torch.zeros(INPUT_SIZE)
        self.stds = torch.ones(INPUT_SIZE)
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if not self.model_path.exists() or not self.normalization_path.exists():
            return

        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        normalization = torch.load(self.normalization_path, map_location=self.device)
        self.means = normalization["means"]
        self.stds = normalization["stds"]

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.means.to(self.device)) / self.stds.to(self.device)

    def _prepare_features(
        self,
        fighter1_stats: list,
        fighter2_stats: list,
    ) -> torch.Tensor:
        features = torch.tensor(
            fighter1_stats + fighter2_stats,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return self._normalize(features)

    def predict(
        self,
        fighter1_stats: list,
        fighter2_stats: list,
    ) -> dict[str, float]:
        """
        Return win / loss / draw probabilities for fighter 1.
        """
        self.model.eval()
        features = self._prepare_features(fighter1_stats, fighter2_stats)

        with torch.no_grad():
            logits = self.model(features)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        return {
            LABEL_COLUMNS[index]: probabilities[index].item()
            for index in range(NUM_CLASSES)
        }

    def predict_fighters(
        self,
        data: Data,
        fighter1: str,
        fighter2: str,
        date: str,
        min_fights: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Build feature vectors from fighter names and return outcome probabilities.
        """
        if min_fights is None:
            min_fights = MIN_FIGHTS

        fighter1_stats = data.find_fighter_stats(fighter1, date, min_fights=min_fights)
        fighter2_stats = data.find_fighter_stats(fighter2, date, min_fights=min_fights)
        return self.predict(fighter1_stats, fighter2_stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fight outcomes.")
    parser.add_argument(
        "--model",
        default="linear",
        choices=sorted(MODELS),
        help="model architecture to load (default: linear)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = FightPredictor(resolve_model(args.model, MODELS))
    data = Data()

    break_works = [
        "",
        "exit",
        "quit",
        "q",
    ]

    while True:
        fighter1 = input("Enter the name of the first fighter: ")
        if fighter1.lower() in break_works:
            break

        fighter2 = input("Enter the name of the second fighter: ")
        if fighter2.lower() in break_works:
            break

        print(predictor.predict_fighters(data, fighter1, fighter2, str(date.today())))
