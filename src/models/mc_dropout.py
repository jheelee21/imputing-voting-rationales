"""
MC Dropout Neural Network for multi-label classification with uncertainty.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.base_model import BaseRationaleModel


class MCDropoutNetwork(nn.Module):
    """Neural network with dropout for uncertainty estimation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        layers = []
        layer_dims = [input_dim] + hidden_dims

        for i in range(len(layer_dims) - 1):
            layers.extend(
                [
                    nn.Linear(layer_dims[i], layer_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )

        layers.append(nn.Linear(layer_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def enable_dropout(self):
        """Enable dropout for MC sampling at test time."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


class MCDropoutModel(BaseRationaleModel):
    """Monte Carlo Dropout model for multi-label classification."""

    def __init__(
        self,
        rationales: List[str],
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_epochs: int = 100,
        batch_size: int = 256,
        num_samples: int = 50,
        weight_decay: float = 1e-4,
        device: str = None,
        random_seed: int = 21,
    ):
        super().__init__(rationales, "mc_dropout", random_seed)

        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.weight_decay = weight_decay

        # Set device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.training_losses = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Train MC Dropout model.
        If mask is provided (bool, shape = y.shape), loss is computed only on observed
        labels (for partial/missing labels). This correctly models rationale prediction
        conditional on dissent, without treating missing rationales as negative.
        """
        input_dim = X.shape[1]
        output_dim = y.shape[1]

        # Create model
        self.model = MCDropoutNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        # Convert to tensors
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.FloatTensor(y).to(self.device)
        mask_tensor = (
            torch.BoolTensor(mask).to(self.device) if mask is not None else None
        )

        # Create data loader (include mask in dataset when present)
        if mask_tensor is not None:
            train_dataset = TensorDataset(X_train, y_train, mask_tensor)
        else:
            train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        if verbose:
            print(f"Training MC Dropout: {input_dim} features → {output_dim} outputs")
            print(
                f"Architecture: {input_dim} → {' → '.join(map(str, self.hidden_dims))} → {output_dim}"
            )
            print(f"Device: {self.device}")
            if mask_tensor is not None:
                obs_frac = mask_tensor.float().mean().item()
                print(
                    f"Partial labels: {obs_frac:.1%} of label matrix observed (masked loss)"
                )

        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_loss_terms = 0

            for batch in train_loader:
                if mask_tensor is not None:
                    batch_x, batch_y, batch_mask = batch
                else:
                    batch_x, batch_y = batch
                    batch_mask = None

                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss_per_element = self.criterion(logits, batch_y)

                if batch_mask is not None:
                    loss = (
                        loss_per_element * batch_mask.float()
                    ).sum() / batch_mask.float().sum().clamp(min=1)
                else:
                    loss = loss_per_element.mean()

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_loss_terms += 1

            avg_loss = epoch_loss / n_loss_terms
            self.training_losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        if verbose:
            print(f"Training completed. Final loss: {self.training_losses[-1]:.4f}")

        return self

    def predict_proba(
        self,
        X: np.ndarray,
        num_samples: Optional[int] = None,
        return_std: bool = False,
    ) -> np.ndarray:
        """Predict probabilities with MC Dropout."""
        if num_samples is None:
            num_samples = self.num_samples

        X_tensor = torch.FloatTensor(X).to(self.device)

        # Enable dropout for MC sampling
        self.model.eval()
        self.model.enable_dropout()

        # Collect predictions from multiple forward passes
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.model(X_tensor)
                probs = torch.sigmoid(logits)
                predictions.append(probs.cpu().numpy())

        predictions = np.array(predictions)
        mean_probs = predictions.mean(axis=0)

        if return_std:
            std_probs = predictions.std(axis=0)
            return mean_probs, std_probs

        return mean_probs

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.

        Returns:
            mean_probs: Mean predicted probabilities
            epistemic_unc: Epistemic uncertainty (model uncertainty)
            total_unc: Total uncertainty
        """
        mean_probs, std_probs = self.predict_proba(X, num_samples, return_std=True)

        epistemic_uncertainty = std_probs
        aleatoric_uncertainty = np.sqrt(mean_probs * (1 - mean_probs))
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)

        return mean_probs, epistemic_uncertainty, total_uncertainty

    def save(self, filepath: str):
        """Save model with PyTorch state dict."""
        save_dict = {
            "rationales": self.rationales,
            "model_type": self.model_type,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "feature_names": self.feature_names,
            "model_state": self.model.state_dict() if self.model else None,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "num_samples": self.num_samples,
                "weight_decay": self.weight_decay,
                "random_seed": self.random_seed,
            },
            "training_losses": self.training_losses,
        }

        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: str = None):
        """Load model from disk."""
        import pickle

        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)

        # Recreate model
        model = cls(
            rationales=save_dict["rationales"],
            hidden_dims=save_dict["hidden_dims"],
            dropout_rate=save_dict["dropout_rate"],
            device=device,
            **save_dict["hyperparameters"],
        )

        model.feature_names = save_dict["feature_names"]
        model.training_losses = save_dict.get("training_losses", [])

        # Recreate network and load weights
        if save_dict["model_state"]:
            input_dim = list(save_dict["model_state"].keys())[0]
            input_dim = save_dict["model_state"][input_dim].shape[1]
            output_dim = len(save_dict["rationales"])

            model.model = MCDropoutNetwork(
                input_dim=input_dim,
                hidden_dims=model.hidden_dims,
                output_dim=output_dim,
                dropout_rate=model.dropout_rate,
            ).to(model.device)

            model.model.load_state_dict(save_dict["model_state"])
            model.is_fitted = True

        print(f"Model loaded from {filepath}")
        return model
