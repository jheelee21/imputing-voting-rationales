import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple
from pathlib import Path
import sys

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro import poutine

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.base_model import BaseRationaleModel


# ============================================================
# Pyro Hierarchical Logistic Module
# ============================================================

class BayesianHierarchicalLogistic(PyroModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_investors: int,
        n_firms: int,
        n_years: int,
        prior_scale: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_investors = n_investors
        self.n_firms = n_firms
        self.n_years = n_years

        # -----------------------------
        # Fixed Effects
        # -----------------------------
        self.beta = PyroSample(
            dist.Normal(0.0, prior_scale)
            .expand([output_dim, input_dim])
            .to_event(2)
        )

        # -----------------------------
        # Hyperpriors for random effects
        # -----------------------------
        self.sigma_investor = PyroSample(dist.HalfNormal(1.0))
        self.sigma_firm = PyroSample(dist.HalfNormal(1.0))
        self.sigma_year = PyroSample(dist.HalfNormal(1.0))

        # -----------------------------
        # Random intercepts
        # -----------------------------
        self.alpha_investor = PyroSample(
            dist.Normal(0.0, self.sigma_investor)
            .expand([output_dim, n_investors])
            .to_event(2)
        )

        self.gamma_firm = PyroSample(
            dist.Normal(0.0, self.sigma_firm)
            .expand([output_dim, n_firms])
            .to_event(2)
        )

        self.delta_year = PyroSample(
            dist.Normal(0.0, self.sigma_year)
            .expand([output_dim, n_years])
            .to_event(2)
        )

    def forward(self, x, investor_idx, firm_idx, year_idx, y=None):

        # Fixed effects
        logits_fixed = x @ self.beta.T  # (batch, output_dim)

        # Random effects
        investor_effect = self.alpha_investor[:, investor_idx].T
        firm_effect = self.gamma_firm[:, firm_idx].T
        year_effect = self.delta_year[:, year_idx].T

        logits = logits_fixed + investor_effect + firm_effect + year_effect

        with pyro.plate("data", x.shape[0]):
            pyro.sample(
                "obs",
                dist.Bernoulli(logits=logits).to_event(1),
                obs=y,
            )

        return logits


# ============================================================
# Main Model Wrapper (Consistent with BNNModel)
# ============================================================

class HierarchicalModel(BaseRationaleModel):

    def __init__(
        self,
        rationales: List[str],
        prior_scale: float = 0.1,
        learning_rate: float = 0.001,
        num_epochs: int = 100,
        batch_size: int = 512,
        num_samples: int = 100,
        patience: int = 10,
        min_delta: float = 1.0,
        grad_clip: float = 1.0,
        device: str = None,
        random_seed: int = 21,
    ):
        super().__init__(rationales, "hierarchical", random_seed)

        self.prior_scale = prior_scale
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        pyro.set_rng_seed(random_seed)

        self.model = None
        self.guide = None
        self.svi = None
        self.training_losses = []
        self.validation_losses = []

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        investor_idx: np.ndarray,
        firm_idx: np.ndarray,
        year_idx: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        investor_val: Optional[np.ndarray] = None,
        firm_val: Optional[np.ndarray] = None,
        year_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):

        input_dim = X.shape[1]
        output_dim = y.shape[1]

        n_investors = investor_idx.max() + 1
        n_firms = firm_idx.max() + 1
        n_years = year_idx.max() + 1

        self.model = BayesianHierarchicalLogistic(
            input_dim=input_dim,
            output_dim=output_dim,
            n_investors=n_investors,
            n_firms=n_firms,
            n_years=n_years,
            prior_scale=self.prior_scale,
        ).to(self.device)

        self.guide = pyro.infer.autoguide.AutoDiagonalNormal(
            poutine.block(self.model, hide=["obs"])
        )

        optimizer = ClippedAdam(
            {"lr": self.learning_rate, "clip_norm": self.grad_clip}
        )

        self.svi = SVI(
            self.model,
            self.guide,
            optimizer,
            loss=Trace_ELBO(num_particles=1),
        )

        # Convert tensors
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        inv_t = torch.LongTensor(investor_idx).to(self.device)
        firm_t = torch.LongTensor(firm_idx).to(self.device)
        year_t = torch.LongTensor(year_idx).to(self.device)

        dataset = TensorDataset(X_t, y_t, inv_t, firm_t, year_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        pyro.clear_param_store()
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):

            epoch_loss = 0
            n_batches = 0

            for bx, by, bi, bf, byear in loader:
                loss = (
                    self.svi.step(bx, bi, bf, byear, by) / len(bx)
                )
                epoch_loss += loss
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            self.training_losses.append(avg_train_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.num_epochs}, "
                    f"Loss: {avg_train_loss:.4f}"
                )

        self.is_fitted = True
        return self

    # ========================================================
    # PREDICT
    # ========================================================

    def predict_proba(
        self,
        X: np.ndarray,
        investor_idx: np.ndarray,
        firm_idx: np.ndarray,
        year_idx: np.ndarray,
        num_samples: Optional[int] = None,
    ):

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        if num_samples is None:
            num_samples = self.num_samples

        X_t = torch.FloatTensor(X).to(self.device)
        inv_t = torch.LongTensor(investor_idx).to(self.device)
        firm_t = torch.LongTensor(firm_idx).to(self.device)
        year_t = torch.LongTensor(year_idx).to(self.device)

        all_probs = []

        with torch.no_grad():
            for _ in range(num_samples):

                guide_trace = pyro.poutine.trace(self.guide).get_trace(
                    X_t, inv_t, firm_t, year_t, None
                )

                model_trace = pyro.poutine.trace(
                    pyro.poutine.replay(self.model, trace=guide_trace)
                ).get_trace(X_t, inv_t, firm_t, year_t, None)

                logits = model_trace.nodes["_RETURN"]["value"]
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        all_probs = np.array(all_probs)
        mean_probs = all_probs.mean(axis=0)

        return mean_probs
