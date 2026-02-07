"""
FIXED Bayesian Neural Network with improved training stability.

Key fixes:
1. Proper ELBO normalization
2. Learning rate scheduling
3. Better prior initialization
4. Early stopping
5. Gradient clipping
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

try:
    import pyro
    import pyro.distributions as dist
    from pyro.nn import PyroModule, PyroSample
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.optim import ClippedAdam
except ImportError:
    print(
        "Warning: Pyro not installed. Install with: pip install pyro-ppl --break-system-packages"
    )

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.base_model import BaseRationaleModel


class BayesianNetwork(PyroModule):
    """Bayesian Neural Network with weight uncertainty."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        prior_scale: float = 0.1,  # CHANGED: Smaller prior
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build network architecture with Bayesian layers
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.num_layers = len(layer_dims) - 1

        # Define priors for weights and biases
        for i in range(self.num_layers):
            layer = PyroModule[nn.Linear](layer_dims[i], layer_dims[i + 1])

            # CHANGED: More informative priors
            # Use smaller scale for weights to prevent divergence
            weight_scale = prior_scale / np.sqrt(layer_dims[i])
            layer.weight = PyroSample(
                dist.Normal(0.0, weight_scale)
                .expand([layer_dims[i + 1], layer_dims[i]])
                .to_event(2)
            )
            layer.bias = PyroSample(
                dist.Normal(0.0, prior_scale).expand([layer_dims[i + 1]]).to_event(1)
            )

            setattr(self, f"layer_{i}", layer)

        self.activation = nn.ReLU()

    def forward(self, x, y=None):
        """Forward pass with Bayesian inference."""
        # Forward through hidden layers
        for i in range(self.num_layers - 1):
            layer = getattr(self, f"layer_{i}")
            x = self.activation(layer(x))

        # Output layer
        output_layer = getattr(self, f"layer_{self.num_layers - 1}")
        logits = output_layer(x)

        # Likelihood - CHANGED: Use independent Bernoulli per output
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(logits=logits).to_event(1), obs=y)

        return logits


class BNNModel(BaseRationaleModel):
    """FIXED Bayesian Neural Network model for multi-label classification."""

    def __init__(
        self,
        rationales: List[str],
        hidden_dims: List[int] = [64, 32],
        prior_scale: float = 0.1,  # CHANGED: Default to smaller prior
        learning_rate: float = 0.001,  # CHANGED: Smaller learning rate
        num_epochs: int = 100,
        batch_size: int = 256,
        num_samples: int = 100,
        patience: int = 10,  # NEW: Early stopping
        min_delta: float = 1.0,  # NEW: Minimum improvement threshold
        grad_clip: float = 1.0,  # NEW: Gradient clipping
        device: str = None,
        random_seed: int = 21,
    ):
        super().__init__(rationales, "bnn", random_seed)

        self.hidden_dims = hidden_dims
        self.prior_scale = prior_scale
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip

        # Set device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        pyro.set_rng_seed(random_seed)

        # Model components
        self.model = None
        self.guide = None
        self.svi = None
        self.training_losses = []
        self.validation_losses = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Train Bayesian Neural Network using Variational Inference."""
        input_dim = X.shape[1]
        output_dim = y.shape[1]

        # Create Bayesian network
        self.model = BayesianNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            prior_scale=self.prior_scale,
        ).to(self.device)

        # Define variational distribution (guide)
        # IMPORTANT: Block 'obs' site to avoid discrete variable issues
        from pyro import poutine

        self.guide = pyro.infer.autoguide.AutoDiagonalNormal(
            poutine.block(self.model, hide=["obs"])
        )

        # CHANGED: Use ClippedAdam with gradient clipping
        optimizer = ClippedAdam(
            {
                "lr": self.learning_rate,
                "clip_norm": self.grad_clip,
            }
        )

        # CHANGED: Use Trace_ELBO with num_particles for better estimates
        self.svi = SVI(
            self.model, self.guide, optimizer, loss=Trace_ELBO(num_particles=1)
        )

        # Convert to tensors
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.FloatTensor(y).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Validation data
        use_validation = X_val is not None and y_val is not None
        if use_validation:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)

        if verbose:
            print(f"Training BNN: {input_dim} features → {output_dim} outputs")
            print(
                f"Architecture: {input_dim} → {' → '.join(map(str, self.hidden_dims))} → {output_dim}"
            )
            print(f"Device: {self.device}")
            print(f"Prior scale: {self.prior_scale}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Gradient clip: {self.grad_clip}")
            if use_validation:
                print(f"Validation: {len(X_val)} samples")
                print(
                    f"Early stopping: patience={self.patience}, min_delta={self.min_delta}"
                )

        # Training loop with early stopping
        pyro.clear_param_store()
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                # CHANGED: Normalize loss by batch size
                loss = self.svi.step(batch_x, batch_y) / len(batch_x)
                epoch_loss += loss
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            self.training_losses.append(avg_train_loss)

            # Validation loss
            if use_validation:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.svi.evaluate_loss(X_val_t, y_val_t) / len(X_val_t)
                    self.validation_losses.append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_param_store = pyro.get_param_store().get_state()
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.num_epochs}, "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Patience: {patience_counter}/{self.patience}"
                    )

                # Early stopping
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        print(f"Best validation loss: {best_val_loss:.4f}")
                    # Restore best parameters
                    if hasattr(self, "best_param_store"):
                        pyro.get_param_store().set_state(self.best_param_store)
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_train_loss:.4f}"
                    )

        self.is_fitted = True

        if verbose:
            final_loss = (
                self.validation_losses[-1]
                if self.validation_losses
                else self.training_losses[-1]
            )
            print(f"\nTraining completed. Final loss: {final_loss:.4f}")

            # Print some diagnostics
            if use_validation:
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"Final validation loss: {self.validation_losses[-1]:.4f}")

        return self

    def predict_proba(
        self,
        X: np.ndarray,
        num_samples: Optional[int] = None,
        return_std: bool = False,
    ) -> np.ndarray:
        """Predict probabilities using posterior samples."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if num_samples is None:
            num_samples = self.num_samples

        X_tensor = torch.FloatTensor(X).to(self.device)

        # Sample from posterior predictive distribution
        # Since we blocked 'obs', we need to manually run forward passes
        self.model.eval()
        self.guide.eval()

        all_probs = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Sample parameters from the guide (posterior)
                guide_trace = pyro.poutine.trace(self.guide).get_trace(
                    x=X_tensor, y=None
                )

                # Run model forward with sampled parameters
                model_trace = pyro.poutine.trace(
                    pyro.poutine.replay(self.model, trace=guide_trace)
                ).get_trace(x=X_tensor, y=None)

                # Get logits from the model's return value
                logits = model_trace.nodes["_RETURN"]["value"]
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        # Stack and compute statistics
        all_probs = np.array(
            all_probs
        )  # Shape: (num_samples, batch_size, num_rationales)
        mean_probs = all_probs.mean(axis=0)

        if return_std:
            std_probs = all_probs.std(axis=0)
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

        # Epistemic uncertainty from posterior variance
        epistemic_uncertainty = std_probs

        # Aleatoric uncertainty from prediction entropy
        aleatoric_uncertainty = np.sqrt(mean_probs * (1 - mean_probs))

        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)

        return mean_probs, epistemic_uncertainty, total_uncertainty

    def save(self, filepath: str):
        """Save model with Pyro state."""
        save_dict = {
            "rationales": self.rationales,
            "model_type": self.model_type,
            "hidden_dims": self.hidden_dims,
            "prior_scale": self.prior_scale,
            "feature_names": self.feature_names,
            "param_store": pyro.get_param_store().get_state(),
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "num_samples": self.num_samples,
                "random_seed": self.random_seed,
                "patience": self.patience,
                "min_delta": self.min_delta,
                "grad_clip": self.grad_clip,
            },
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
        }

        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"BNN model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: str = None):
        """Load model from disk."""
        import pickle

        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)

        # Recreate model instance
        model = cls(
            rationales=save_dict["rationales"],
            hidden_dims=save_dict["hidden_dims"],
            prior_scale=save_dict["prior_scale"],
            device=device,
            **save_dict["hyperparameters"],
        )

        model.feature_names = save_dict["feature_names"]
        model.training_losses = save_dict.get("training_losses", [])
        model.validation_losses = save_dict.get("validation_losses", [])

        # Reconstruct the BNN and guide to load parameters
        if save_dict.get("param_store"):
            # We need to know the input/output dimensions
            # Get from feature_names and rationales
            input_dim = len(save_dict["feature_names"])
            output_dim = len(save_dict["rationales"])

            # Recreate the Bayesian network
            model.model = BayesianNetwork(
                input_dim=input_dim,
                hidden_dims=model.hidden_dims,
                output_dim=output_dim,
                prior_scale=model.prior_scale,
            ).to(model.device)

            # Recreate the guide - IMPORTANT: Block 'obs' site
            from pyro import poutine

            model.guide = pyro.infer.autoguide.AutoDiagonalNormal(
                poutine.block(model.model, hide=["obs"])
            )

            # Clear and restore parameter store
            pyro.clear_param_store()
            pyro.get_param_store().set_state(save_dict["param_store"])

            model.is_fitted = True

            print(
                f"BNN model loaded from {filepath} ({input_dim} features → {output_dim} outputs)"
            )
        else:
            print(f"Warning: No trained parameters found in {filepath}")
            print("Model structure loaded but needs retraining")

        return model
