"""
Bayesian Neural Network using Pyro for multi-label classification with uncertainty.
Provides more principled uncertainty quantification than MC Dropout.
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
    from pyro.optim import Adam
except ImportError:
    print("Warning: Pyro not installed. Install with: pip install pyro-ppl --break-system-packages")

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.base_model import BaseRationaleModel


class BayesianNetwork(PyroModule):
    """Bayesian Neural Network with weight uncertainty."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        prior_scale: float = 1.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network architecture with Bayesian layers
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.num_layers = len(layer_dims) - 1
        
        # Define priors for weights and biases with unique names per layer
        # Use setattr to create uniquely named attributes
        for i in range(self.num_layers):
            layer = PyroModule[nn.Linear](layer_dims[i], layer_dims[i + 1])
            
            # Set priors on weights and biases
            layer.weight = PyroSample(
                dist.Normal(0., prior_scale).expand([layer_dims[i + 1], layer_dims[i]]).to_event(2)
            )
            layer.bias = PyroSample(
                dist.Normal(0., prior_scale).expand([layer_dims[i + 1]]).to_event(1)
            )
            
            # Register with unique name using setattr
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
        
        # Likelihood
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(logits=logits).to_event(1), obs=y)
        
        return logits


class BNNModel(BaseRationaleModel):
    """Bayesian Neural Network model for multi-label classification."""
    
    def __init__(
        self,
        rationales: List[str],
        hidden_dims: List[int] = [64, 32],
        prior_scale: float = 1.0,
        learning_rate: float = 0.01,
        num_epochs: int = 100,
        batch_size: int = 256,
        num_samples: int = 100,
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
        
        # Set device
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.model)
        
        # Setup optimizer and inference
        optimizer = Adam({"lr": self.learning_rate})
        self.svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        # Convert to tensors
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.FloatTensor(y).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if verbose:
            print(f"Training BNN: {input_dim} features → {output_dim} outputs")
            print(f"Architecture: {input_dim} → {' → '.join(map(str, self.hidden_dims))} → {output_dim}")
            print(f"Device: {self.device}")
            print(f"Prior scale: {self.prior_scale}")
        
        # Training loop
        pyro.clear_param_store()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                loss = self.svi.step(batch_x, batch_y)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(train_loader)
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
        """Predict probabilities using posterior samples."""
        if num_samples is None:
            num_samples = self.num_samples
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Sample from posterior
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
        
        with torch.no_grad():
            samples = predictive(X_tensor)
            logits = samples['_RETURN']  # Get the logits from forward pass
            probs = torch.sigmoid(logits)
        
        # Convert to numpy
        probs_np = probs.cpu().numpy()
        mean_probs = probs_np.mean(axis=0)
        
        if return_std:
            std_probs = probs_np.std(axis=0)
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
            'rationales': self.rationales,
            'model_type': self.model_type,
            'hidden_dims': self.hidden_dims,
            'prior_scale': self.prior_scale,
            'feature_names': self.feature_names,
            'param_store': pyro.get_param_store().get_state(),
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'num_samples': self.num_samples,
                'random_seed': self.random_seed,
            },
            'training_losses': self.training_losses,
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"BNN model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: str = None):
        """Load model from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Recreate model
        model = cls(
            rationales=save_dict['rationales'],
            hidden_dims=save_dict['hidden_dims'],
            prior_scale=save_dict['prior_scale'],
            device=device,
            **save_dict['hyperparameters']
        )
        
        model.feature_names = save_dict['feature_names']
        model.training_losses = save_dict.get('training_losses', [])
        
        # Recreate network and guide
        if save_dict['param_store']:
            # Would need to recreate the exact architecture and reload params
            # This is complex with Pyro - simplified for now
            print("Warning: Full model loading not implemented - retrain required")
        
        print(f"BNN model loaded from {filepath}")
        return model