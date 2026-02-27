import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple
from pathlib import Path
import sys
import pickle

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from pyro import poutine

sys.path.append(str(Path(__file__).parent.parent))
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
        prior_scale: float = 1.0,
        n_random_slope_features: int = 5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_investors = n_investors
        self.n_firms = n_firms
        self.n_years = n_years
        self.n_slope_feats = min(n_random_slope_features, input_dim)
        self.class_weights = class_weights  # shape: (output_dim,)

        # -----------------------------------------------------------
        # 1.  Rationale hierarchy on fixed effects
        #     beta[r, j] ~ Normal(global_mean[j], sigma_rationale)
        #     This pools information across rationales so that rare ones
        #     (e.g. combined_ceo_chairman) borrow strength from common ones.
        # -----------------------------------------------------------
        self.global_beta_mean = PyroSample(
            dist.Normal(0.0, prior_scale).expand([input_dim]).to_event(1)
        )
        self.sigma_rationale = PyroSample(dist.HalfNormal(1.0))

        self.beta = PyroSample(
            lambda self: dist.Normal(
                self.global_beta_mean.unsqueeze(0).expand(output_dim, input_dim),
                self.sigma_rationale,
            ).to_event(2)
        )

        # Per-rationale intercept (bias)
        self.intercept = PyroSample(
            dist.Normal(0.0, prior_scale).expand([output_dim]).to_event(1)
        )

        # -----------------------------------------------------------
        # 2.  Random intercepts with learned hyperpriors
        # -----------------------------------------------------------
        self.sigma_investor = PyroSample(dist.HalfNormal(1.0))
        self.sigma_firm = PyroSample(dist.HalfNormal(1.0))
        self.sigma_year = PyroSample(dist.HalfNormal(1.0))

        self.alpha_investor = PyroSample(
            lambda self: dist.Normal(0.0, self.sigma_investor)
            .expand([output_dim, n_investors])
            .to_event(2)
        )
        self.gamma_firm = PyroSample(
            lambda self: dist.Normal(0.0, self.sigma_firm)
            .expand([output_dim, n_firms])
            .to_event(2)
        )
        self.delta_year = PyroSample(
            lambda self: dist.Normal(0.0, self.sigma_year)
            .expand([output_dim, n_years])
            .to_event(2)
        )

        # -----------------------------------------------------------
        # 3.  Investor random slopes (improvement #2)
        #     Allows each investor to have their own sensitivity to the
        #     most domain-critical predictors (e.g. Per_female, AvTenure).
        #     Applied to the first `n_slope_feats` columns of X, which
        #     should be ordered with key predictors first in DataManager.
        # -----------------------------------------------------------
        if self.n_slope_feats > 0:
            self.sigma_slope = PyroSample(dist.HalfNormal(0.5))
            self.random_slope = PyroSample(
                lambda self: dist.Normal(0.0, self.sigma_slope)
                .expand([output_dim, n_investors, self.n_slope_feats])
                .to_event(3)
            )

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------
    def forward(self, x, investor_idx, firm_idx, year_idx, y=None):
        batch = x.shape[0]

        # Fixed effects: (batch, output_dim)
        logits = x @ self.beta.T + self.intercept.unsqueeze(0)

        # Random intercepts
        logits = logits + self.alpha_investor[:, investor_idx].T
        logits = logits + self.gamma_firm[:, firm_idx].T
        logits = logits + self.delta_year[:, year_idx].T

        # Random slopes on first n_slope_feats predictors
        if self.n_slope_feats > 0:
            # random_slope: (output_dim, n_investors, n_slope_feats)
            # gather investor-specific slopes → (batch, output_dim, n_slope_feats)
            slopes = self.random_slope[
                :, investor_idx, :
            ]  # (output_dim, batch, n_slope_feats)
            slopes = slopes.permute(1, 0, 2)  # (batch, output_dim, n_slope_feats)
            x_key = x[:, : self.n_slope_feats]  # (batch, n_slope_feats)
            # einsum: for each obs and rationale, dot product with key features
            logits = logits + torch.einsum("brf,bf->br", slopes, x_key)

        with pyro.plate("data", batch):
            if y is not None:
                if self.class_weights is not None:
                    # Improvement #5: correct per-sample factor inside plate
                    # log_prob: (batch, output_dim)
                    log_prob = dist.Bernoulli(logits=logits).log_prob(y)
                    weights = torch.where(
                        y == 1,
                        self.class_weights,
                        torch.ones_like(self.class_weights),
                    )
                    # Sum over output_dim → (batch,) so plate handles batch scaling
                    per_sample_lp = (log_prob * weights).sum(-1)
                    pyro.factor("weighted_obs", per_sample_lp)
                else:
                    pyro.sample(
                        "obs",
                        dist.Bernoulli(logits=logits).to_event(1),
                        obs=y,
                    )

        return logits


# ============================================================
# Main Model Wrapper
# ============================================================


class HierarchicalModel(BaseRationaleModel):
    """
    Bayesian hierarchical model wrapper.

    Key improvements vs original:
      - AutoLowRankMultivariateNormal guide (rank configurable)
      - AUC-based early stopping on validation data
      - Investor random slopes on top-K features
      - Cosine LR annealing
      - predict_with_uncertainty() for epistemic uncertainty
      - 300 posterior samples by default
    """

    def __init__(
        self,
        rationales: List[str],
        prior_scale: float = 1.0,
        learning_rate: float = 0.005,
        num_epochs: int = 300,
        batch_size: int = 256,
        num_samples: int = 300,  # Improvement #8: more samples
        patience: int = 20,
        min_delta: float = 0.001,  # Improvement #9: tighter delta
        grad_clip: float = 5.0,
        lrk_rank: int = 10,  # Improvement #1: low-rank guide
        n_random_slope_features: int = 5,  # Improvement #2: random slopes
        eval_interval: int = 5,  # AUC eval every N epochs
        device: str = None,
        random_seed: int = 123,
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
        self.lrk_rank = lrk_rank
        self.n_random_slope_features = n_random_slope_features
        self.eval_interval = eval_interval

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        pyro.set_rng_seed(random_seed)

        self.model = None
        self.guide = None
        self.svi = None
        self.training_losses: List[float] = []
        self.validation_losses: List[float] = []
        self.validation_aucs: List[float] = []

    # ============================================================
    # FIT
    # ============================================================

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

        n_investors = int(investor_idx.max()) + 1
        n_firms = int(firm_idx.max()) + 1
        n_years = int(year_idx.max()) + 1

        # Class weights: neg/pos ratio per rationale
        pos_counts = y.sum(axis=0)
        neg_counts = y.shape[0] - pos_counts
        class_weights = torch.FloatTensor(neg_counts / (pos_counts + 1e-8)).to(
            self.device
        )

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Training HIERARCHICAL BAYESIAN MODEL")
            print(f"  Rationales       : {self.rationales}")
            print(f"  Features         : {input_dim}")
            print(f"  Train samples    : {X.shape[0]:,}")
            print(f"  Investors/Firms/Yrs: {n_investors}/{n_firms}/{n_years}")
            print(f"  Random slopes    : first {self.n_random_slope_features} features")
            print(f"  Guide rank       : {self.lrk_rank}")
            print(f"  Device           : {self.device}")
            print(f"  Class weights    : {class_weights.cpu().numpy().round(2)}")
            print(f"{'=' * 70}\n")

        self.model = BayesianHierarchicalLogistic(
            input_dim=input_dim,
            output_dim=output_dim,
            n_investors=n_investors,
            n_firms=n_firms,
            n_years=n_years,
            prior_scale=self.prior_scale,
            n_random_slope_features=self.n_random_slope_features,
            class_weights=class_weights,
        ).to(self.device)

        # -----------------------------------------------------------
        # Improvement #1: AutoLowRankMultivariateNormal
        #   Captures posterior correlations between parameters while
        #   remaining cheaper than full multivariate normal.
        # -----------------------------------------------------------
        self.guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(
            poutine.block(self.model, hide=["obs", "weighted_obs"]),
            rank=self.lrk_rank,
        )

        optimizer = ClippedAdam({"lr": self.learning_rate, "clip_norm": self.grad_clip})

        # Improvement #10: more particles → lower-variance ELBO gradient
        self.svi = SVI(
            self.model,
            self.guide,
            optimizer,
            loss=Trace_ELBO(num_particles=4),
        )

        # ---- tensors ----
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        inv_t = torch.LongTensor(investor_idx).to(self.device)
        firm_t = torch.LongTensor(firm_idx).to(self.device)
        year_t = torch.LongTensor(year_idx).to(self.device)

        dataset = TensorDataset(X_t, y_t, inv_t, firm_t, year_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        use_val = all(
            v is not None for v in [X_val, y_val, investor_val, firm_val, year_val]
        )
        if use_val:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)
            inv_val_t = torch.LongTensor(investor_val).to(self.device)
            firm_val_t = torch.LongTensor(firm_val).to(self.device)
            year_val_t = torch.LongTensor(year_val).to(self.device)

        # ---- cosine LR schedule ----
        # Decay lr from initial to 10% of initial over all epochs
        def _cosine_lr(epoch: int) -> float:
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * epoch / self.num_epochs))

        pyro.clear_param_store()
        best_metric = -np.inf  # AUC (higher = better)
        best_param_store = None
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # ---- cosine LR annealing (improvement #7) ----
            current_lr = self.learning_rate * _cosine_lr(epoch)
            optimizer.set_state({"lr": current_lr, "clip_norm": self.grad_clip})

            # ---- training pass ----
            epoch_loss = 0.0
            n_batches = 0
            for bx, by, bi, bf, byear in loader:
                loss = self.svi.step(bx, bi, bf, byear, by) / len(bx)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)

            # ---- validation & early stopping (improvement #3) ----
            if use_val and (epoch + 1) % self.eval_interval == 0:
                val_elbo = self.svi.evaluate_loss(
                    X_val_t, inv_val_t, firm_val_t, year_val_t, y_val_t
                ) / len(X_val_t)
                self.validation_losses.append(val_elbo)

                # AUC-based stopping metric
                mean_auc = self._compute_val_auc(
                    X_val_t, inv_val_t, firm_val_t, year_val_t, y_val
                )
                self.validation_aucs.append(mean_auc)

                if verbose and (epoch + 1) % (self.eval_interval * 2) == 0:
                    print(
                        f"Epoch {epoch + 1:4d}/{self.num_epochs}  "
                        f"lr={current_lr:.5f}  "
                        f"train_loss={avg_loss:.4f}  "
                        f"val_elbo={val_elbo:.4f}  "
                        f"val_auc={mean_auc:.4f}  "
                        f"patience={patience_counter}/{self.patience}"
                    )

                if mean_auc > best_metric + self.min_delta:
                    best_metric = mean_auc
                    patience_counter = 0
                    best_param_store = pyro.get_param_store().get_state()
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(
                            f"\n  Early stopping at epoch {epoch + 1} "
                            f"(best val AUC={best_metric:.4f})"
                        )
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0 and not use_val:
                    print(
                        f"Epoch {epoch + 1:4d}/{self.num_epochs}  "
                        f"lr={current_lr:.5f}  "
                        f"train_loss={avg_loss:.4f}"
                    )

        # Restore best parameters if we used validation
        if use_val and best_param_store is not None:
            pyro.get_param_store().set_state(best_param_store)
            if verbose:
                print(f"\n  Best model restored (val AUC={best_metric:.4f})")

        self.is_fitted = True
        return self

    # ============================================================
    # Helpers
    # ============================================================

    def _compute_val_auc(self, X_t, inv_t, firm_t, year_t, y_val_np):
        """Compute mean AUC across rationales on validation data (fast, 50 samples)."""
        from sklearn.metrics import roc_auc_score

        with torch.no_grad():
            probs = self._sample_probs(X_t, inv_t, firm_t, year_t, n_samples=50)
            mean_probs = probs.mean(0).cpu().numpy()  # (n_val, output_dim)

        aucs = []
        for j in range(y_val_np.shape[1]):
            labels = y_val_np[:, j]
            if labels.sum() > 0 and (1 - labels).sum() > 0:
                try:
                    aucs.append(roc_auc_score(labels, mean_probs[:, j]))
                except Exception:
                    pass

        return float(np.mean(aucs)) if aucs else 0.5

    def _sample_probs(self, X_t, inv_t, firm_t, year_t, n_samples: int):
        """
        Draw n_samples from the posterior predictive and return sigmoid probabilities.
        Returns tensor of shape (n_samples, batch, output_dim).
        """
        predictive = Predictive(
            self.model,
            guide=self.guide,
            num_samples=n_samples,
            return_sites=["_RETURN"],
        )
        with torch.no_grad():
            samples = predictive(X_t, inv_t, firm_t, year_t, None)

        logits = samples["_RETURN"]  # (n_samples, batch, output_dim)
        return torch.sigmoid(logits)

    # ============================================================
    # PREDICT
    # ============================================================

    def predict_proba(
        self,
        X: np.ndarray,
        investor_idx: np.ndarray,
        firm_idx: np.ndarray,
        year_idx: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns mean predicted probability (n_obs, output_dim).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        n_samples = num_samples or self.num_samples
        X_t = torch.FloatTensor(X).to(self.device)
        inv_t = torch.LongTensor(investor_idx).to(self.device)
        firm_t = torch.LongTensor(firm_idx).to(self.device)
        year_t = torch.LongTensor(year_idx).to(self.device)

        probs = self._sample_probs(X_t, inv_t, firm_t, year_t, n_samples)
        return probs.mean(0).cpu().numpy()

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        investor_idx: np.ndarray,
        firm_idx: np.ndarray,
        year_idx: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean_probs, epistemic_std) each of shape (n_obs, output_dim).

        epistemic_std is the standard deviation across posterior samples —
        observations where this is high are those the model is uncertain about.
        This is useful for active learning (query the most uncertain cases).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        n_samples = num_samples or self.num_samples
        X_t = torch.FloatTensor(X).to(self.device)
        inv_t = torch.LongTensor(investor_idx).to(self.device)
        firm_t = torch.LongTensor(firm_idx).to(self.device)
        year_t = torch.LongTensor(year_idx).to(self.device)

        probs = self._sample_probs(X_t, inv_t, firm_t, year_t, n_samples)
        # probs: (n_samples, batch, output_dim)
        mean_probs = probs.mean(0).cpu().numpy()
        epistemic_std = probs.std(0).cpu().numpy()
        return mean_probs, epistemic_std

    # ============================================================
    # CALIBRATION DIAGNOSTICS
    # ============================================================

    def calibration_summary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        investor_idx: np.ndarray,
        firm_idx: np.ndarray,
        year_idx: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """
        Compares predicted mean probabilities to empirical label rates in
        equal-width probability bins.  Returns a dict with per-rationale ECE
        (Expected Calibration Error) and a summary DataFrame.

        Usage:
            cal = model.calibration_summary(X_val, y_val, inv, firm, yr)
            print(cal["ece"])
        """
        import pandas as pd

        probs = self.predict_proba(X, investor_idx, firm_idx, year_idx)
        bins = np.linspace(0, 1, n_bins + 1)
        results = {}
        rows = []

        for j, rat in enumerate(self.rationales):
            p = probs[:, j]
            labels = y[:, j]
            ece = 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (p >= lo) & (p < hi)
                if mask.sum() == 0:
                    continue
                frac_pos = labels[mask].mean()
                mean_pred = p[mask].mean()
                weight = mask.sum() / len(p)
                ece += weight * abs(mean_pred - frac_pos)
                rows.append(
                    {
                        "rationale": rat,
                        "bin_lo": lo,
                        "bin_hi": hi,
                        "mean_pred": mean_pred,
                        "frac_pos": frac_pos,
                        "n": mask.sum(),
                    }
                )
            results[rat] = round(ece, 4)

        print("\nExpected Calibration Error (ECE) per rationale:")
        for rat, ece in results.items():
            print(f"  {rat:30s}: {ece:.4f}")

        return {"ece": results, "detail": pd.DataFrame(rows)}

    # ============================================================
    # SAVE / LOAD
    # ============================================================

    def save(self, filepath: str):
        save_dict = {
            "rationales": self.rationales,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "param_store": pyro.get_param_store().get_state(),
            "hyperparameters": {
                "prior_scale": self.prior_scale,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "num_samples": self.num_samples,
                "patience": self.patience,
                "min_delta": self.min_delta,
                "grad_clip": self.grad_clip,
                "lrk_rank": self.lrk_rank,
                "n_random_slope_features": self.n_random_slope_features,
                "eval_interval": self.eval_interval,
                "random_seed": self.random_seed,
            },
            "model_config": {
                "input_dim": self.model.input_dim if self.model else None,
                "output_dim": self.model.output_dim if self.model else None,
                "n_investors": self.model.n_investors if self.model else None,
                "n_firms": self.model.n_firms if self.model else None,
                "n_years": self.model.n_years if self.model else None,
                "n_slope_feats": self.model.n_slope_feats if self.model else None,
                "class_weights": (
                    self.model.class_weights.cpu().numpy()
                    if self.model and self.model.class_weights is not None
                    else None
                ),
            },
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "validation_aucs": self.validation_aucs,
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"HierarchicalModel saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: str = None):
        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)

        hp = save_dict["hyperparameters"]
        model = cls(
            rationales=save_dict["rationales"],
            device=device,
            **hp,
        )
        model.feature_names = save_dict.get("feature_names")
        model.training_losses = save_dict.get("training_losses", [])
        model.validation_losses = save_dict.get("validation_losses", [])
        model.validation_aucs = save_dict.get("validation_aucs", [])

        cfg = save_dict.get("model_config", {})
        if cfg.get("input_dim") and save_dict.get("param_store"):
            cw = cfg.get("class_weights")
            class_weights = (
                torch.FloatTensor(cw).to(model.device) if cw is not None else None
            )

            model.model = BayesianHierarchicalLogistic(
                input_dim=cfg["input_dim"],
                output_dim=cfg["output_dim"],
                n_investors=cfg["n_investors"],
                n_firms=cfg["n_firms"],
                n_years=cfg["n_years"],
                prior_scale=hp["prior_scale"],
                n_random_slope_features=cfg.get(
                    "n_slope_feats", hp.get("n_random_slope_features", 5)
                ),
                class_weights=class_weights,
            ).to(model.device)

            model.guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(
                poutine.block(model.model, hide=["obs", "weighted_obs"]),
                rank=hp["lrk_rank"],
            )

            pyro.clear_param_store()
            pyro.get_param_store().set_state(save_dict["param_store"])

            # Initialise guide so parameters are registered before use
            dummy_x = torch.zeros(2, cfg["input_dim"]).to(model.device)
            dummy_idx = torch.zeros(2, dtype=torch.long).to(model.device)
            model.guide(dummy_x, dummy_idx, dummy_idx, dummy_idx, None)

            model.is_fitted = True
            print(
                f"HierarchicalModel loaded from {filepath} "
                f"({cfg['input_dim']} features → {cfg['output_dim']} rationales)"
            )
        else:
            print(f"Warning: Incomplete saved state in {filepath}. Needs retraining.")

        return model
