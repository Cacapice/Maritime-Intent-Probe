"""
sae.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
======
SparseAutoencoder: decomposes polysemantic residual stream vectors into
sparse monosemantic features before probing.

Following Bricken et al. (2023) "Towards Monosemanticity" and
Templeton et al. (2024) "Scaling Monosemanticity".

To substitute a different SAE architecture (e.g. TopK, Gated):
  - Subclass SparseAutoencoder or replace this file.
  - The interface that experiment.py and monitor.py depend on is:
      sae.encode(x: Tensor) → Tensor
      sae.fit(activations, n_steps, batch_size, lr) → list[float]
      sae.state_dict() / sae.load_state_dict()  (inherited from nn.Module)
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    """
    Single-layer SAE following Bricken et al. (2023) / Templeton et al. (2024).
    Decomposes polysemantic residual stream vectors into a sparse set of
    monosemantic features before probing, so that positive probe results
    can be attributed to specific interpretable directions rather than
    opaque mixtures in the raw hidden state.

    Architecture: x → encoder (ReLU + L1) → sparse features → decoder → x̂
    Loss: reconstruction MSE + λ * L1 sparsity penalty.

    Args:
        d_model:    Residual stream dimension (e.g. 4096)
        n_features: Dictionary size — typically 4–16× d_model
        l1_coeff:   Sparsity penalty weight (tune to target ~1–5% feature density)
    """

    def __init__(self, d_model: int, n_features: int, l1_coeff: float = 2e-4):
        super().__init__()
        self.encoder  = nn.Linear(d_model, n_features, bias=True)
        self.decoder  = nn.Linear(n_features, d_model, bias=True)
        self.l1_coeff = l1_coeff
        # Normalise decoder columns to unit norm at init (Bricken et al.)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (features, reconstruction, loss)."""
        x = x.to(self.encoder.weight.dtype)   # SAE params are fp32; tolerate a bf16 stream
        features       = torch.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        loss = (
            (reconstruction - x).pow(2).mean()        # reconstruction
            + self.l1_coeff * features.abs().mean()    # sparsity
        )
        return features, reconstruction, loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without computing loss — used during collection and runtime."""
        p = next(self.parameters())
        x = x.to(device=p.device, dtype=p.dtype)   # match params (fp32) — tolerate bf16 stream
        return torch.relu(self.encoder(x))

    def fit(
        self,
        activations: torch.Tensor,      # [n_samples, d_model]
        n_steps: int = 2000,
        batch_size: int = 256,
        lr: float = 1e-4,
        labels: torch.Tensor | None = None,   # [n_samples] int — for per-class MSE logging
    ) -> dict:
        """
        Train SAE on collected activations.

        Returns a results dict containing:
          losses: list[float] — per-step loss values
          reconstruction_mse: float — final-step mean MSE across all samples
          per_class_mse: dict[int, float] | None — per-label reconstruction MSE
            if labels are provided (Bias Control 4 — Rajamanoharan et al. 2024 /
            Smith et al. 2025). Log this and flag any class where MSE exceeds
            1.5× the global mean — probe AUC results for flagged classes should
            be reported with a reconstruction quality caveat.

        Activations are moved to the SAE's device so training runs on GPU
        when available — critical for Colab iteration speed. Place the SAE
        on device before calling fit():
            sae = SparseAutoencoder(d_model, n_features).to(device)
            sae.fit(activations, labels=labels)

        Iteration guidance:
          n_steps=500   — fast scan across layer subset
          n_steps=2000  — full fit once best layer is identified
        """
        device      = next(self.parameters()).device
        activations = activations.to(device)
        opt         = torch.optim.Adam(self.parameters(), lr=lr)
        losses      = []

        for step in range(n_steps):
            idx   = torch.randint(0, activations.size(0), (batch_size,), device=device)
            batch = activations[idx]
            _, _, loss = self(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Re-normalise decoder columns AFTER the optimiser step so the next
            # forward pass sees unit-norm dictionary directions (Bricken et al.
            # 2023). Doing this before opt.step() would let the step move the
            # columns straight back off the unit sphere.
            with torch.no_grad():
                self.decoder.weight.data = nn.functional.normalize(
                    self.decoder.weight.data, dim=0
                )
            losses.append(loss.item())
            if (step + 1) % 500 == 0:
                logger.info("SAE step %d/%d | loss %.4f", step + 1, n_steps, loss.item())

        # ── Per-class reconstruction error (Bias Control 4) ───────────────────
        per_class_mse: dict[int, float] | None = None
        with torch.no_grad():
            _, recon, _ = self(activations)
            mse_all     = (recon - activations).pow(2).mean(dim=1)   # [n_samples]
            global_mse  = mse_all.mean().item()

            if labels is not None:
                labels_dev = labels.to(device)
                class_ids  = labels_dev.unique().tolist()
                per_class_mse = {}
                threshold  = 1.5 * global_mse
                for cls in class_ids:
                    mask    = labels_dev == int(cls)
                    cls_mse = mse_all[mask].mean().item()
                    per_class_mse[int(cls)] = round(cls_mse, 6)
                    if cls_mse > threshold:
                        logger.warning(
                            "Bias Control 4: class %d reconstruction MSE %.4f "
                            "exceeds 1.5× global mean (%.4f). Probe AUC for "
                            "this class should be reported with a reconstruction "
                            "quality caveat (Smith et al. 2025).",
                            int(cls), cls_mse, global_mse,
                        )
                    else:
                        logger.info(
                            "Reconstruction MSE class %d: %.4f (global mean: %.4f)",
                            int(cls), cls_mse, global_mse,
                        )

        # ── SAE health diagnostics (secondary; Smith et al. 2025 context) ─────
        # Reported alongside per-class MSE so SAE-space probe results can be read
        # against dictionary quality. Not a pre-registered gate; pure reporting.
        with torch.no_grad():
            feats, recon, _ = self(activations)
            l0          = feats.gt(0).float().sum(dim=1).mean().item()   # mean active features/sample
            n_features  = feats.shape[1]
            alive       = feats.gt(0).any(dim=0)                          # any activation across samples
            dead_frac   = 1.0 - (alive.float().mean().item())
            # fraction-of-variance-unexplained -> explained variance
            resid_var   = (activations - recon).pow(2).sum().item()
            total_var   = (activations - activations.mean(dim=0, keepdim=True)).pow(2).sum().item()
            explained_var = 1.0 - (resid_var / total_var) if total_var > 0 else float("nan")
        health = {
            "l0_sparsity":       round(l0, 3),                  # mean active features per sample
            "l0_fraction":       round(l0 / n_features, 4),
            "dead_feature_frac": round(dead_frac, 4),
            "explained_variance": round(explained_var, 4),
            "n_features":        int(n_features),
        }
        logger.info(
            "SAE health: L0=%.1f (%.1f%% of %d feats), dead=%.1f%%, explained var=%.3f",
            l0, 100 * l0 / n_features, n_features, 100 * dead_frac, explained_var,
        )

        return {
            "losses":           losses,
            "reconstruction_mse": round(global_mse, 6),
            "per_class_mse":    per_class_mse,
            "health":           health,
        }
