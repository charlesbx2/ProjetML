#!/usr/bin/env python

import os
import random
import itertools
from datetime import datetime
from os.path import join as pjoin
from params import n_harmonics

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sympy.combinatorics import Permutation
from params import periodic
if periodic:
    from params import a, n_cells, V0


# ---------------------------------------------------------------------------
# Couche gaussienne (cas non-périodique)
# ---------------------------------------------------------------------------

class GaussianLayer(nn.Module):

    def __init__(self, n_particles: int, dim_physical: int, sigma_init: float = 2.0):
        super().__init__()
        self.n_particles  = n_particles
        self.dim_physical = dim_physical
        self.log_sigma    = nn.Parameter(torch.tensor(float(np.log(sigma_init))))

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r          = x.reshape(-1, self.n_particles, self.dim_physical)
        gauss      = torch.exp(-(r ** 2) / (2.0 * self.sigma))
        per_particle = gauss.prod(dim=2)
        return per_particle.prod(dim=1)


# ---------------------------------------------------------------------------
# Embedding de Fourier (cas périodique)
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """
    Encode x → [1, cos(2πx/a), sin(2πx/a), cos(4πx/a), sin(4πx/a), ...]

    Périodicité exacte par construction : embed(x + a) = embed(x).
    Les dérivées sont analytiquement connues et bien contrôlées.

    output_dim = 2 * n_harmonics + 1
    """

    def __init__(self, a: float, n_harmonics: int = 8):
        super().__init__()
        self.a           = a
        self.n_harmonics = n_harmonics
        ns = torch.arange(1, n_harmonics + 1).float()
        self.register_buffer('ns', ns)   # pas entraînable

    @property
    def output_dim(self) -> int:
        return 2 * self.n_harmonics + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (batch, 1)
        sortie : (batch, 2*n_harmonics + 1)
        """
        angles = 2 * torch.pi * x * self.ns / self.a   # (batch, n_harmonics)
        ones   = torch.ones(x.shape[0], 1, device=x.device)
        return torch.cat([ones, torch.cos(angles), torch.sin(angles)], dim=1)


# ---------------------------------------------------------------------------
# BlochNet : réseau pour u_k avec embedding de Fourier
# ---------------------------------------------------------------------------

class BlochNet(nn.Module):
    """
    Réseau pour la partie périodique u_k de la fonction de Bloch.

    Architecture :
        x (réel) → FourierEmbedding → MLP → u_k(x) réel

    Propriétés :
    - Périodicité exacte par construction (via l'embedding)
    - Dérivées bien contrôlées (pas d'oscillations parasites)
    - Pas besoin de x % a ni de translate_u
    """

    def __init__(self,
                 a:            float,
                 n_harmonics:  int   = 8,
                 n_layers:     int   = 2,
                 layer_size:   int   = 128,
                 reg:          float = 1e-8):
        super().__init__()
        self.reg     = reg
        self.fourier = FourierEmbedding(a, n_harmonics)

        input_dim   = self.fourier.output_dim
        layers_list = [nn.Linear(input_dim, layer_size), nn.GELU()]
        for _ in range(n_layers):
            layers_list += [nn.Linear(layer_size, layer_size), nn.GELU()]
        layers_list.append(nn.Linear(layer_size, 1))
        self.net = nn.Sequential(*layers_list)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (batch, 1)
        sortie : (batch,) — u_k(x) réel
        """
        emb = self.fourier(x)          # (batch, 2*n_harmonics+1)
        return self.net(emb).squeeze(-1)

    def l2_loss(self) -> torch.Tensor:
        l2 = torch.tensor(0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                l2 = l2 + (m.weight ** 2).sum()
        return self.reg * l2


class BlochNetComplex(nn.Module):
    """Deux BlochNet pour Re et Im."""
    def __init__(self, a, n_harmonics=10, n_layers=2, layer_size=64):
        super().__init__()
        self.net_real = BlochNet(a, n_harmonics, n_layers, layer_size)
        self.net_imag = BlochNet(a, n_harmonics, n_layers, layer_size)

    def forward(self, x):
        return self.net_real(x), self.net_imag(x)


class WavefunctionNet(nn.Module):

    def __init__(self,
                 dim:          int,
                 n_particles:  int,
                 dim_physical: int,
                 n_layers:     int,
                 layer_size:   int,
                 reg:          float = 1e-8,
                 sigma_init:   float = 2.0):
        super().__init__()
        self.reg = reg

        layers_list = [nn.Linear(dim, layer_size), nn.GELU()]
        for _ in range(n_layers):
            layers_list += [nn.Linear(layer_size, layer_size), nn.GELU()]
        self.hidden       = nn.Sequential(*layers_list)
        self.output_layer = nn.Linear(layer_size, 1)
        self.boundary     = GaussianLayer(n_particles, dim_physical, sigma_init)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h        = self.hidden(x)
        out      = self.output_layer(h).squeeze(-1)
        boundary = self.boundary(x)
        return out * boundary

    def l2_loss(self) -> torch.Tensor:
        l2 = torch.tensor(0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                l2 = l2 + (m.weight ** 2).sum()
        return self.reg * l2


class WavefunctionNetComplex(nn.Module):
    def __init__(self, dim, n_particles, dim_physical, n_layers, layer_size,
                 reg=1e-8):
        super().__init__()
        self.net_real = WavefunctionNet(dim, n_particles, dim_physical,
                                        n_layers, layer_size, reg)
        self.net_imag = WavefunctionNet(dim, n_particles, dim_physical,
                                        n_layers, layer_size, reg)

    def forward(self, x):
        return self.net_real(x) + 1j * self.net_imag(x)




# ---------------------------------------------------------------------------
# neural_fit : entraînement + retour des callables fitfunc / d2_fitfunc
# ---------------------------------------------------------------------------

def neural_fit(x:            np.ndarray,
               psi:          np.ndarray,
               n_samples:    int,
               perm_subset:  int,
               perm:         list,
               parity:       list,
               analysis_data:dict,
               iteration:    int,
               load_weights: int,
               bosonic:      bool,
               U:            float,
               n_particles:  int,
               dim_physical: int,
               n_layers:     int,
               layer_size:   int,
               epochs:       int,
               batch_size:   int,
               reg:          float,
               normalize:    bool  = True,
               device:       str   = None,
               # Paramètres spécifiques au cas périodique
               n_harmonics:  int   = 8):


    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    dim        = x.shape[1]
    is_complex = np.iscomplexobj(psi)

    # --- Modèle ---
    if periodic:
        # BlochNet : x ∈ [0, a], u_k réel
        model = BlochNet(a=a, n_harmonics=n_harmonics,
                         n_layers=n_layers, layer_size=layer_size,
                         reg=reg).to(dev)
    elif is_complex:
        model = WavefunctionNetComplex(dim, n_particles, dim_physical,
                                       n_layers, layer_size, reg).to(dev)
    else:
        model = WavefunctionNet(dim, n_particles, dim_physical,
                                n_layers, layer_size, reg).to(dev)

    # --- Chemins de sauvegarde ---
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = pjoin(checkpoint_dir, f"wf_checkpoint_{iteration}.pt")

    if load_weights == 1:
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state)
        print(f"Poids chargés depuis {ckpt_path}")
    else:
        # --- Préparation des données ---
        if periodic:
            # Replie x dans [0, a] avant d'entraîner
            x_folded = x % a                       # (N, 1)
            x_t = torch.tensor(x_folded, dtype=torch.float32, device=dev)
            y_t = torch.tensor(psi.ravel(), dtype=torch.float32, device=dev)
        else:
            x_t = torch.tensor(x, dtype=torch.float32, device=dev)
            if is_complex:
                y_t = torch.tensor(psi.ravel(),
                                   dtype=torch.complex64, device=dev)
            else:
                y_t = torch.tensor(psi.ravel(),
                                   dtype=torch.float32, device=dev)

        dataset    = TensorDataset(x_t, y_t)
        val_size   = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        # --- Optimiseur ---
        if periodic:
            '''
            optimizer = optim.Adam(model.parameters(),
                                   lr=1e-3,
                                   weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
            '''
            optimizer = optim.SGD(model.parameters(),
                                  lr=0.05,
                                  momentum=0.9,
                                  weight_decay=1e-5,
                                  nesterov=False)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=0.05,
                                  momentum=0.9,
                                  weight_decay=1e-5,
                                  nesterov=False)
            scheduler = None

        def mse_loss(pred, target):
            if torch.is_complex(target):
                return (torch.abs(pred - target) ** 2).mean()
            return nn.functional.mse_loss(pred, target)

        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

        print("Entraînement en cours...")
        model.train()
        for epoch in range(epochs):
            train_loss, train_mae, n_train = 0.0, 0.0, 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                if is_complex and not periodic:
                    pred = model.net_real(xb) + 1j * model.net_imag(xb)
                else:
                    pred = model(xb)
                loss = mse_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
                train_mae  += (pred - yb).abs().mean().item() * len(xb)
                n_train    += len(xb)

            model.eval()
            val_loss, val_mae, n_val = 0.0, 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    if is_complex and not periodic:
                        pred = model.net_real(xb) + 1j * model.net_imag(xb)
                    else:
                        pred = model(xb)
                    loss = mse_loss(pred, yb)
                    val_loss += loss.item() * len(xb)
                    val_mae  += (pred - yb).abs().mean().item() * len(xb)
                    n_val    += len(xb)
            model.train()

            if scheduler is not None:
                scheduler.step(val_loss / n_val)

            history['loss'].append(train_loss / n_train)
            history['val_loss'].append(val_loss / n_val)
            history['mae'].append(train_mae / n_train)
            history['val_mae'].append(val_mae / n_val)

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  "
                      f"loss={history['loss'][-1]:.4e}  "
                      f"val_loss={history['val_loss'][-1]:.4e}")

        analysis_data['history'] = history
        torch.save(model.state_dict(), ckpt_path)
        print(f"Poids sauvegardés dans {ckpt_path}")

    model.eval()

    # -----------------------------------------------------------------------
    # fitfunc : évalue u_k (périodique) ou ψ (non-périodique)
    # -----------------------------------------------------------------------
    def fitfunc(x_np: np.ndarray, batch_size_inf: int = 128) -> np.ndarray:
        results = []
        if periodic:
            # Replie dans [0, a] avant d'évaluer
            x_in = torch.tensor((x_np % a), dtype=torch.float32, device=dev)
        else:
            x_in = torch.tensor(x_np, dtype=torch.float32, device=dev)

        with torch.no_grad():
            for i in range(0, len(x_in), batch_size_inf):
                xb  = x_in[i:i + batch_size_inf]
                if is_complex and not periodic:
                    out = (model.net_real(xb) + 1j * model.net_imag(xb)).cpu()
                    results.append(out.numpy().astype(np.complex64))
                else:
                    out = model(xb).cpu()
                    results.append(out.numpy())
        return np.concatenate(results, axis=0).reshape(-1, 1)


    def d2_fitfunc(x_input) -> np.ndarray:
        if isinstance(x_input, np.ndarray):
            if periodic:
                x_t = torch.tensor((x_input % a),
                                   dtype=torch.float32, device=dev)
            else:
                x_t = torch.tensor(x_input, dtype=torch.float32, device=dev)
        else:
            x_t = x_input.to(dev)
            if periodic:
                x_t = x_t % a

        x_t = x_t.detach().requires_grad_(True)

        def compute_laplacian(scalar_field):
            grad1 = torch.autograd.grad(
                outputs=scalar_field,
                inputs=x_t,
                grad_outputs=torch.ones_like(scalar_field),
                create_graph=True,
                retain_graph=True
            )[0]
            laplacian = torch.zeros(x_t.shape[0], device=dev)
            for i in range(x_t.shape[1]):
                grad2_i = torch.autograd.grad(
                    outputs=grad1[:, i],
                    inputs=x_t,
                    grad_outputs=torch.ones(x_t.shape[0], device=dev),
                    retain_graph=(i < x_t.shape[1] - 1),
                    create_graph=False
                )[0]
                laplacian += grad2_i[:, i]
            return laplacian

        if periodic:
            u_val     = model(x_t)
            lap       = compute_laplacian(u_val)
            lap_np    = lap.cpu().detach().numpy()

        elif is_complex:
            psi_real  = model.net_real(x_t)
            psi_imag  = model.net_imag(x_t)
            lap_real  = compute_laplacian(psi_real)
            lap_imag  = compute_laplacian(psi_imag)
            lap_c     = lap_real + 1j * lap_imag
            psi_abs   = (psi_real.detach()**2 + psi_imag.detach()**2).sqrt() + 1e-10
            lap_abs   = torch.abs(lap_c)
            scale     = torch.clamp(lap_abs, max=1e3 * psi_abs) / (lap_abs + 1e-10)
            lap_c     = lap_c * scale
            lap_np    = lap_c.cpu().detach().numpy().astype(np.complex64)

        else:
            psi_val   = model(x_t)
            lap       = compute_laplacian(psi_val)
            psi_abs   = psi_val.detach().abs() + 1e-10
            lap       = torch.clamp(lap, -1e3 * psi_abs, 1e3 * psi_abs)
            lap_np    = lap.cpu().detach().numpy()

        return lap_np.reshape(-1, 1)

    return fitfunc, d2_fitfunc


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from sample_distribution_Nd import sample_mixed
    import harmonic_oscillator_Nd as ho

    n_particles  = 2
    dim_physical = 2
    dim          = n_particles * dim_physical
    nsamples     = 500
    epochs       = 20
    batch_size   = 64
    n_layers     = 1
    layer_size   = 64
    reg          = 1e-8
    bosonic      = False
    perm_subset  = 2
    U            = 0.0
    nu           = 0.001
    hbar = m = omega = 1.0
    offset       = [-1.0, 1.0]
    xmax         = 5.0

    def wavefunction(x):
        return ho.eigenfunction_samples(0, x, n_particles, dim_physical,
                                        offset, hbar, m, omega, bosonic)

    def P0(x):
        return np.real(np.conj(wavefunction(x)) * wavefunction(x))

    x0      = np.zeros(dim)
    step    = np.full(dim, xmax)
    samples = sample_mixed(P0, x0, step, xmax, nsamples, 5, 0.2)

    psi_vals = wavefunction(samples).real
    psi_vals = psi_vals.reshape(-1, 1) / np.max(np.abs(psi_vals))

    from itertools import permutations as iperms
    perm   = list(iperms(range(n_particles)))
    parity = [[Permutation(list(p)).parity()] for p in perm]
    subset = sorted(random.sample(range(len(parity)), perm_subset))
    perm   = [perm[i] for i in subset]
    parity = [parity[i] for i in subset]

    analysis_data = {}
    fitfunc, d2_fitfunc = neural_fit(
        samples, psi_vals, nsamples, perm_subset, perm, parity,
        analysis_data, 0, 0, bosonic, U,
        n_particles, dim_physical, n_layers, layer_size, epochs, batch_size, reg
    )

    psi_pred = fitfunc(samples[:10])
    print("ψ prédit :", psi_pred)
    lap = d2_fitfunc(samples[:10])
    print("∇²ψ      :", lap)
    print("Test OK")
