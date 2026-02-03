"""
MC Dropout Neural Network for Multi-Label Classification with Uncertainty

This provides an alternative to full Bayesian inference using Monte Carlo Dropout.
It's simpler, faster, and often more stable while still providing uncertainty estimates.

References:
- Gal & Ghahramani (2016): Dropout as a Bayesian Approximation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


class MCDropoutNetwork(nn.Module):
    """
    Neural Network with MC Dropout for uncertainty estimation.
    
    Uses dropout at inference time to sample different network configurations,
    approximating a Bayesian posterior.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        layer_dims = [input_dim] + hidden_dims
        
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(layer_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass with dropout enabled"""
        return self.network(x)
    
    def enable_dropout(self):
        """Enable dropout for MC sampling at test time"""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


class MCDropoutModel:
    """
    Monte Carlo Dropout model for multi-label classification with uncertainty.
    
    Simpler and more stable alternative to full Bayesian inference.
    """
    
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
        """
        Initialize MC Dropout Model.
        
        Args:
            rationales: List of rationale labels to predict
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability (0.1-0.3 recommended)
            learning_rate: Learning rate for Adam optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            num_samples: Number of MC samples for prediction
            weight_decay: L2 regularization strength
            device: Device to run on ('cpu' or 'cuda')
            random_seed: Random seed for reproducibility
        """
        self.rationales = rationales
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.input_dim = None
        self.feature_names = None
        self.training_losses = []
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Train the MC Dropout Neural Network.
        
        Args:
            X: Training features [n_samples, n_features]
            y: Training labels [n_samples, n_labels]
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress
        """
        self.input_dim = X.shape[1]
        output_dim = y.shape[1]
        
        # Create model
        self.model = MCDropoutNetwork(
            input_dim=self.input_dim,
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
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.FloatTensor(y).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        if verbose:
            print(f"Training MC Dropout NN with {self.input_dim} features, {output_dim} outputs")
            print(f"Architecture: {self.input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> {output_dim}")
            print(f"Dropout rate: {self.dropout_rate}")
            print(f"Device: {self.device}")
            print(f"Training samples: {len(X)}, Positive rates: {y.mean(axis=0)}")
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.training_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
                
                # Validation metrics if provided
                if X_val is not None and y_val is not None:
                    val_metrics = self._compute_validation_metrics(X_val, y_val)
                    print(f"  Val - Avg F1: {val_metrics['avg_f1']:.4f}, "
                          f"Avg AUC: {val_metrics['avg_auc']:.4f}")
        
        if verbose:
            print(f"\nTraining completed. Final loss: {self.training_losses[-1]:.4f}")
        
        return self
    
    def predict_proba(
        self,
        X: np.ndarray,
        num_samples: Optional[int] = None,
        return_std: bool = False,
    ) -> np.ndarray:
        """
        Predict class probabilities with MC Dropout uncertainty estimation.
        
        Args:
            X: Input features [n_samples, n_features]
            num_samples: Number of MC samples (default: self.num_samples)
            return_std: If True, also return standard deviation of predictions
        
        Returns:
            If return_std=False: probabilities [n_samples, n_labels]
            If return_std=True: (probabilities, std_dev) tuple
        """
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
        
        predictions = np.array(predictions)  # [num_samples, batch_size, num_labels]
        
        # Compute mean and std
        mean_probs = predictions.mean(axis=0)
        
        if return_std:
            std_probs = predictions.std(axis=0)
            return mean_probs, std_probs
        
        return mean_probs
    
    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
        num_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Input features
            threshold: Decision threshold
            num_samples: Number of MC samples
        
        Returns:
            Binary predictions [n_samples, n_labels]
        """
        probs = self.predict_proba(X, num_samples=num_samples)
        return (probs >= threshold).astype(int)
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with epistemic and total uncertainty estimates.
        
        Args:
            X: Input features
            num_samples: Number of MC samples
        
        Returns:
            Tuple of (mean_probs, epistemic_uncertainty, total_uncertainty)
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        mean_probs, std_probs = self.predict_proba(
            X, num_samples=num_samples, return_std=True
        )
        
        # Epistemic uncertainty (model uncertainty from dropout)
        epistemic_uncertainty = std_probs
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric_uncertainty = np.sqrt(mean_probs * (1 - mean_probs))
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        return mean_probs, epistemic_uncertainty, total_uncertainty
    
    def _compute_validation_metrics(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Compute validation metrics."""
        from sklearn.metrics import f1_score, roc_auc_score
        
        y_pred = self.predict(X_val)
        y_prob = self.predict_proba(X_val)
        
        metrics = {
            'avg_f1': f1_score(y_val, y_pred, average='macro', zero_division=0),
        }
        
        try:
            metrics['avg_auc'] = roc_auc_score(y_val, y_prob, average='macro')
        except:
            metrics['avg_auc'] = 0.0
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        save_dict = {
            'rationales': self.rationales,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'input_dim': self.input_dim,
            'feature_names': self.feature_names,
            'model_state': self.model.state_dict() if self.model else None,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'num_samples': self.num_samples,
                'weight_decay': self.weight_decay,
                'random_seed': self.random_seed,
            },
            'training_losses': self.training_losses,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = None):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Recreate model
        model = cls(
            rationales=save_dict['rationales'],
            hidden_dims=save_dict['hidden_dims'],
            dropout_rate=save_dict['dropout_rate'],
            device=device,
            **save_dict['hyperparameters']
        )
        
        model.input_dim = save_dict['input_dim']
        model.feature_names = save_dict['feature_names']
        model.training_losses = save_dict.get('training_losses', [])
        
        # Recreate network
        output_dim = len(save_dict['rationales'])
        model.model = MCDropoutNetwork(
            input_dim=model.input_dim,
            hidden_dims=model.hidden_dims,
            output_dim=output_dim,
            dropout_rate=model.dropout_rate,
        ).to(model.device)
        
        # Load state dict
        if save_dict['model_state']:
            model.model.load_state_dict(save_dict['model_state'])
        
        print(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Example usage
    print("MC Dropout Model for Multi-Label Classification")
    print("=" * 70)
    
    # Simulate data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_labels = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples, n_labels) > 0.7).astype(int)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Label prevalence: {y.mean(axis=0)}")
    
    # Train model
    model = MCDropoutModel(
        rationales=['rat1', 'rat2', 'rat3', 'rat4', 'rat5'],
        hidden_dims=[64, 32],
        dropout_rate=0.2,
        num_epochs=50,
        batch_size=64,
    )
    
    model.fit(X[:800], y[:800], X_val=X[800:], y_val=y[800:])
    
    # Predict with uncertainty
    mean_probs, epistemic_unc, total_unc = model.predict_with_uncertainty(X[800:])
    
    print(f"\nPrediction mean probabilities shape: {mean_probs.shape}")
    print(f"Epistemic uncertainty shape: {epistemic_unc.shape}")
    print(f"Total uncertainty shape: {total_unc.shape}")
    
    print(f"\nAverage epistemic uncertainty: {epistemic_unc.mean():.4f}")
    print(f"Average total uncertainty: {total_unc.mean():.4f}")