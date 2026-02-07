"""
Gaussian Process models for voting rationale prediction.
Includes standard GP, Sparse GP, and Deep Kernel Learning variants.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.base_model import BaseRationaleModel

try:
    import gpytorch
    import torch
    from gpytorch.models import ApproximateGP, VariationalGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        VariationalStrategy,
    )
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
    from gpytorch.likelihoods import BernoulliLikelihood
    from gpytorch.mlls import VariationalELBO

    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    print(
        "Warning: GPyTorch not installed. Install with: pip install gpytorch --break-system-packages"
    )


class VariationalGPClassifier(ApproximateGP):
    """Variational GP for binary classification using inducing points."""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel_type: str = "rbf",
        ard_dims: Optional[int] = None,
    ):
        # Define variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        # Mean and kernel
        self.mean_module = ConstantMean()

        if kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=ard_dims)
        elif kernel_type == "matern":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        self.covar_module = ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepKernelGPClassifier(ApproximateGP):
    """Deep Kernel Learning: Neural network feature extractor + GP."""

    def __init__(
        self,
        input_dim: int,
        inducing_points: torch.Tensor,
        hidden_dims: List[int] = [64, 32],
        kernel_type: str = "rbf",
    ):
        # Define variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        # Neural network feature extractor
        layers = []
        layer_dims = [input_dim] + hidden_dims
        for i in range(len(layer_dims) - 1):
            layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(torch.nn.ReLU())

        self.feature_extractor = torch.nn.Sequential(*layers)

        # Mean and kernel (operating on learned features)
        self.mean_module = ConstantMean()

        if kernel_type == "rbf":
            base_kernel = RBFKernel()
        elif kernel_type == "matern":
            base_kernel = MaternKernel(nu=2.5)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        self.covar_module = ScaleKernel(base_kernel)

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        # GP on features
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(BaseRationaleModel):
    """Gaussian Process model for single-label classification."""

    def __init__(
        self,
        rationale: str,
        model_type: str = "sparse_gp",  # 'sparse_gp' or 'deep_kernel'
        kernel_type: str = "rbf",
        num_inducing: int = 500,
        learning_rate: float = 0.01,
        num_epochs: int = 100,
        batch_size: int = 256,
        hidden_dims: List[int] = [64, 32],  # For deep kernel only
        use_ard: bool = True,
        device: str = None,
        random_seed: int = 21,
    ):
        if not GPYTORCH_AVAILABLE:
            raise ImportError("GPyTorch required for GP models")

        super().__init__([rationale], f"gp_{model_type}", random_seed)

        self.rationale = rationale
        self.gp_model_type = model_type
        self.kernel_type = kernel_type
        self.num_inducing = num_inducing
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.use_ard = use_ard

        # Set device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Model components
        self.model = None
        self.likelihood = None
        self.training_losses = []

    def _initialize_inducing_points(self, X: np.ndarray) -> torch.Tensor:
        """Initialize inducing points using k-means or random selection."""
        n_inducing = min(self.num_inducing, len(X))

        # Random selection for simplicity
        indices = np.random.choice(len(X), n_inducing, replace=False)
        inducing_points = torch.FloatTensor(X[indices]).to(self.device)

        return inducing_points

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Train Gaussian Process model."""
        n_pos = y.sum()

        if n_pos == 0:
            if verbose:
                print(f"No positive samples for '{self.rationale}'. Skipping training.")
            return self

        if verbose:
            print(
                f"Training {self.gp_model_type.upper()} for '{self.rationale}': "
                f"{len(y)} samples, {n_pos} positive ({n_pos / len(y):.2%})"
            )

        # Convert to tensors
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.FloatTensor(y).to(self.device)

        # Initialize inducing points
        inducing_points = self._initialize_inducing_points(X)

        # Create model
        if self.gp_model_type == "sparse_gp":
            ard_dims = X.shape[1] if self.use_ard else None
            self.model = VariationalGPClassifier(
                inducing_points=inducing_points,
                kernel_type=self.kernel_type,
                ard_dims=ard_dims,
            ).to(self.device)

        elif self.gp_model_type == "deep_kernel":
            self.model = DeepKernelGPClassifier(
                input_dim=X.shape[1],
                inducing_points=inducing_points,
                hidden_dims=self.hidden_dims,
                kernel_type=self.kernel_type,
            ).to(self.device)

        else:
            raise ValueError(f"Unknown GP model type: {self.gp_model_type}")

        # Likelihood
        self.likelihood = BernoulliLikelihood().to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.learning_rate,
        )

        # Loss function
        mll = VariationalELBO(self.likelihood, self.model, num_data=len(y_train))

        if verbose:
            print(f"Inducing points: {inducing_points.shape[0]}")
            print(f"Device: {self.device}")

        # Training loop
        self.model.train()
        self.likelihood.train()

        # Create batches
        n_batches = (len(X_train) + self.batch_size - 1) // self.batch_size

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            # Shuffle data
            perm = torch.randperm(len(X_train))

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X_train))
                batch_indices = perm[start_idx:end_idx]

                batch_x = X_train[batch_indices]
                batch_y = y_train[batch_indices]

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = -mll(output, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
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
        return_std: bool = False,
    ) -> np.ndarray:
        """Predict probabilities with GP."""
        if not self.is_fitted:
            return np.zeros(len(X))

        self.model.eval()
        self.likelihood.eval()

        X_test = torch.FloatTensor(X).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get predictive distribution
            output = self.model(X_test)
            pred_dist = self.likelihood(output)

            # Get probabilities
            probs = pred_dist.mean.cpu().numpy()

            if return_std:
                # Uncertainty from GP variance
                std = pred_dist.variance.sqrt().cpu().numpy()
                return probs, std

        return probs

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.

        Returns:
            probs: Predicted probabilities
            uncertainty: Predictive uncertainty (standard deviation)
        """
        probs, std = self.predict_proba(X, return_std=True)
        return probs, std

    def save(self, filepath: str):
        """Save GP model."""
        save_dict = {
            "rationale": self.rationale,
            "model_type": self.model_type,
            "gp_model_type": self.gp_model_type,
            "kernel_type": self.kernel_type,
            "feature_names": self.feature_names,
            "model_state": self.model.state_dict() if self.model else None,
            "likelihood_state": self.likelihood.state_dict()
            if self.likelihood
            else None,
            "hyperparameters": {
                "num_inducing": self.num_inducing,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "hidden_dims": self.hidden_dims,
                "use_ard": self.use_ard,
                "random_seed": self.random_seed,
            },
            "training_losses": self.training_losses,
        }

        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"GP model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: str = None):
        """Load GP model from disk."""
        import pickle

        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)

        model = cls(
            rationale=save_dict["rationale"],
            model_type=save_dict["gp_model_type"],
            kernel_type=save_dict["kernel_type"],
            device=device,
            **save_dict["hyperparameters"],
        )

        model.feature_names = save_dict["feature_names"]
        model.training_losses = save_dict.get("training_losses", [])

        # Note: Full model reconstruction requires recreating exact architecture
        # Simplified for now
        print(
            "Warning: Full GP model loading requires retraining with same inducing points"
        )
        print(f"GP model metadata loaded from {filepath}")

        return model


class MultiLabelGP:
    """
    Multi-label wrapper for Gaussian Process models.
    Trains one GP per rationale.
    """

    def __init__(
        self,
        rationales: List[str],
        model_type: str = "sparse_gp",
        kernel_type: str = "rbf",
        num_inducing: int = 500,
        learning_rate: float = 0.01,
        num_epochs: int = 100,
        batch_size: int = 256,
        hidden_dims: List[int] = [64, 32],
        use_ard: bool = True,
        device: str = None,
        random_seed: int = 21,
    ):
        self.rationales = rationales
        self.model_type = model_type
        self.kernel_type = kernel_type
        self.num_inducing = num_inducing
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.use_ard = use_ard
        self.device = device
        self.random_seed = random_seed

        self.models = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Train GP models for all rationales."""
        for i, rationale in enumerate(self.rationales):
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Training {rationale} ({i + 1}/{len(self.rationales)})")
                print(f"{'=' * 80}")

            model = GPModel(
                rationale=rationale,
                model_type=self.model_type,
                kernel_type=self.kernel_type,
                num_inducing=self.num_inducing,
                learning_rate=self.learning_rate,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                hidden_dims=self.hidden_dims,
                use_ard=self.use_ard,
                device=self.device,
                random_seed=self.random_seed,
            )

            y_single = y[:, i]
            y_val_single = y_val[:, i] if y_val is not None else None

            model.fit(X, y_single, X_val, y_val_single, verbose=verbose)
            self.models[rationale] = model

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for all rationales."""
        probs = []
        for rationale in self.rationales:
            if rationale in self.models:
                probs.append(self.models[rationale].predict_proba(X))
            else:
                probs.append(np.zeros(len(X)))

        return np.column_stack(probs)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels for all rationales."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainties for all rationales."""
        probs_list = []
        uncert_list = []

        for rationale in self.rationales:
            if rationale in self.models:
                probs, uncert = self.models[rationale].predict_with_uncertainty(X)
                probs_list.append(probs)
                uncert_list.append(uncert)
            else:
                probs_list.append(np.zeros(len(X)))
                uncert_list.append(np.ones(len(X)))

        return np.column_stack(probs_list), np.column_stack(uncert_list)
