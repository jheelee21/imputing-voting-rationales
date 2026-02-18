import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from pyro import poutine

from models.base_model import BaseRationaleModel


class ImprovedBayesianHierarchicalLogistic(PyroModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_investors: int,
        n_firms: int,
        n_years: int,
        prior_scale: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_investors = n_investors
        self.n_firms = n_firms
        self.n_years = n_years
        self.beta_loc = torch.zeros(output_dim, input_dim)
        self.beta_scale = torch.ones(output_dim, input_dim) * prior_scale

        self.beta = PyroSample(dist.Normal(self.beta_loc, self.beta_scale).to_event(2))

        self.sigma_investor = PyroSample(dist.HalfNormal(1.0))
        self.sigma_firm = PyroSample(dist.HalfNormal(1.0))
        self.sigma_year = PyroSample(dist.HalfNormal(1.0))

        self.alpha_investor = PyroSample(
            dist.Normal(0.0, self.sigma_investor)
            .expand([output_dim, n_investors])
            .to_event(2)
        )

        self.gamma_firm = PyroSample(
            dist.Normal(0.0, self.sigma_firm).expand([output_dim, n_firms]).to_event(2)
        )

        self.delta_year = PyroSample(
            dist.Normal(0.0, self.sigma_year).expand([output_dim, n_years]).to_event(2)
        )

        self.sigma_slope = PyroSample(dist.HalfNormal(0.5))

        self.random_slope = PyroSample(
            dist.Normal(0.0, self.sigma_slope)
            .expand([output_dim, n_investors, input_dim])
            .to_event(3)
        )

    def forward(self, x, investor_idx, firm_idx, year_idx, y=None):
        logits_fixed = x @ self.beta.T

        investor_intercept = self.alpha_investor[:, investor_idx].T
        firm_intercept = self.gamma_firm[:, firm_idx].T
        year_intercept = self.delta_year[:, year_idx].T

        slopes = self.random_slope[:, investor_idx, :]
        slopes = slopes.permute(1, 0, 2)

        logits_random_slope = torch.einsum("bij,bj->bi", slopes, x)

        logits = (
            logits_fixed
            + investor_intercept
            + firm_intercept
            + year_intercept
            + logits_random_slope
        )

        with pyro.plate("data", x.shape[0]):
            pyro.sample(
                "obs",
                dist.Bernoulli(logits=logits).to_event(1),
                obs=y,
            )

        return logits


class ImprovedHierarchicalModel(BaseRationaleModel):
    def __init__(
        self,
        rationales: List[str],
        prior_scale: float = 0.5,
        learning_rate: float = 0.05,
        num_epochs: int = 150,
        batch_size: int = 512,
        num_samples: int = 100,
        grad_clip: float = 1.0,
        device: str = None,
        random_seed: int = 21,
    ):
        self.rationales = rationales
        self.prior_scale = prior_scale
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
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
        self.is_fitted = False

    # =========================================================
    # FIT
    # =========================================================

    def fit(self, X, y, investor_idx, firm_idx, year_idx):
        input_dim = X.shape[1]
        output_dim = y.shape[1]

        n_investors = investor_idx.max() + 1
        n_firms = firm_idx.max() + 1
        n_years = year_idx.max() + 1

        self.model = ImprovedBayesianHierarchicalLogistic(
            input_dim,
            output_dim,
            n_investors,
            n_firms,
            n_years,
            self.prior_scale,
        ).to(self.device)

        self.guide = pyro.infer.autoguide.AutoMultivariateNormal(
            poutine.block(self.model, hide=["obs"])
        )

        optimizer = ClippedAdam({"lr": self.learning_rate, "clip_norm": self.grad_clip})

        self.svi = SVI(
            self.model,
            self.guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        inv_t = torch.LongTensor(investor_idx).to(self.device)
        firm_t = torch.LongTensor(firm_idx).to(self.device)
        year_t = torch.LongTensor(year_idx).to(self.device)

        dataset = TensorDataset(X_t, y_t, inv_t, firm_t, year_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        pyro.clear_param_store()

        for epoch in range(self.num_epochs):
            epoch_loss = 0

            for bx, by, bi, bf, byear in loader:
                loss = self.svi.step(bx, bi, bf, byear, by)
                epoch_loss += loss

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.2f}")

        self.is_fitted = True
        return self

    # =========================================================
    # PREDICT
    # =========================================================

    def predict_proba(self, X, investor_idx, firm_idx, year_idx):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        X_t = torch.FloatTensor(X).to(self.device)
        inv_t = torch.LongTensor(investor_idx).to(self.device)
        firm_t = torch.LongTensor(firm_idx).to(self.device)
        year_t = torch.LongTensor(year_idx).to(self.device)

        predictive = Predictive(
            self.model,
            guide=self.guide,
            num_samples=self.num_samples,
            return_sites=["_RETURN"],
        )

        samples = predictive(X_t, inv_t, firm_t, year_t, None)
        logits = samples["_RETURN"]

        probs = torch.sigmoid(logits).mean(0)

        return probs.cpu().numpy()
