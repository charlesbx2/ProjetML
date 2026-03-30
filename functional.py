#!/usr/bin/env python
"""
Traduction PyTorch de functional_neural_function_approximation_Nd.py
Auteur original : Atanasova, Bernheimer & Cohen (Nature Communications 2023)
Traduction : projet M2 Physique - Machine Learning

Changements principaux par rapport à la version TensorFlow :
  - GaussianLayer → module nn.Module avec sigma entraînable
  - neural_fit   → boucle d'entraînement manuelle (torch.optim.SGD)
  - d2_fitfunc   → laplacien via torch.autograd (grad de grad)
  - Sauvegarde/chargement via torch.save / model.load_state_dict
"""
   

import os
import random
import itertools
from datetime import datetime
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sympy.combinatorics import Permutation

# ---------------------------------------------------------------------------
# Utilitaire : permutations + augmentation de données
# ---------------------------------------------------------------------------

def permute(x_init: np.ndarray,
            y_init: np.ndarray,
            perm: list,
            parity: list,
            d: int,
            n_particles: int,
            bosonic: bool):
    """
    Génère des échantillons supplémentaires par permutation des particules,
    en respectant la (anti)symétrie bosonique/fermionique.

    Paramètres
    ----------
    x_init     : (N, d*n_particles) coordonnées des échantillons
    y_init     : (N,) ou (N,1) valeurs de la fonction d'onde
    perm       : liste de permutations (tuples)
    parity     : liste des parités associées [[p], ...]
    d          : dimension physique (dim_physical)
    n_particles: nombre de particules
    bosonic    : True → symétrie bosonique, False → antisymétrie fermionique

    Retourne
    --------
    x_perm : (N * len(perm), d*n_particles)
    y_perm : (N * len(perm),)
    """
    x_init = x_init.reshape(x_init.shape[0], n_particles, d)
    num_perm = len(parity)
    x_perm = np.copy(x_init)
    y_perm = np.copy(y_init).ravel()

    for p in range(1, num_perm):
        permutation = list(perm[p])
        x_p = np.take(x_init, permutation, axis=1)
        x_perm = np.append(x_perm, x_p, axis=0)
        sign = 1 if bosonic else (-1) ** parity[p][0]
        y_p = sign * y_perm[:y_init.shape[0]]   # on permute toujours l'original
        y_perm = np.append(y_perm, y_p)

    return x_perm.reshape(x_perm.shape[0], d * n_particles), y_perm


# ---------------------------------------------------------------------------
# Couche gaussienne (condition aux bords)
# ---------------------------------------------------------------------------


class GaussianLayer(nn.Module):
 
    def __init__(self, n_particles: int, dim_physical: int, sigma_init: float = 2.0):
        super().__init__()
        self.n_particles = n_particles
        self.dim_physical = dim_physical
        # sigma entraînable (log-paramétré pour rester positif)
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(sigma_init))))
 
    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)   # garantit sigma > 0
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, n_particles * dim_physical)
        sortie : (batch,)
        """
        r = x.reshape(-1, self.n_particles, self.dim_physical)  # (B, N, d)
        # Gaussienne sur chaque coordonnée
        gauss = torch.exp(-(r ** 2) / (2.0 * self.sigma))       # (B, N, d)
        # Produit sur les coordonnées spatiales → (B, N)
        per_particle = gauss.prod(dim=2)
        # Produit sur les particules → (B,)
        return per_particle.prod(dim=1)
 

# ---------------------------------------------------------------------------
# Architecture principale
# ---------------------------------------------------------------------------

class WavefunctionNet(nn.Module):
    """
    Réseau de neurones dense (MLP) pour représenter une fonction d'onde.

    Architecture :
        input (dim,)
        → Dense(layer_size) + GELU
        → [Dense(layer_size) + GELU] × n_layers
        → Dense(1, linear)          [output_initial]
        × GaussianLayer(input)      [condition aux bords]
        → sortie scalaire (batch,)

    Les poids sont initialisés de façon uniforme (comme dans la version TF).
    """

    def __init__(self,
                 dim: int,
                 n_particles: int,
                 dim_physical: int,
                 n_layers: int,
                 layer_size: int,
                 reg: float = 1e-8,
                 sigma_init: float = 2.0):
        super().__init__()
        self.reg = reg

        # --- couches cachées ---
        layers_list = []
        layers_list.append(nn.Linear(dim, layer_size))
        layers_list.append(nn.GELU())
        for _ in range(n_layers):
            layers_list.append(nn.Linear(layer_size, layer_size))
            layers_list.append(nn.GELU())
        self.hidden = nn.Sequential(*layers_list)

        # --- couche de sortie linéaire ---
        self.output_layer = nn.Linear(layer_size, 1)

        # --- enveloppe gaussienne ---
        self.boundary = GaussianLayer(n_particles, dim_physical, sigma_init)

        # Initialisation uniforme (comme kernel_initializer='uniform' dans TF)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, dim)
        sortie : (batch,) — valeur de la fonction d'onde en chaque point
        """
        h = self.hidden(x)
        out = self.output_layer(h).squeeze(-1)   # (batch,)
        boundary = self.boundary(x)              # (batch,)
        return out * boundary

    def l2_loss(self) -> torch.Tensor:
        """Regularisation L2 sur tous les paramètres linéaires."""
        l2 = torch.tensor(0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                l2 = l2 + (m.weight ** 2).sum()
        return self.reg * l2


# ---------------------------------------------------------------------------
# Fonction principale : entraînement + retour des fonctions psi / Δpsi
# ---------------------------------------------------------------------------

def neural_fit(x: np.ndarray,
               psi: np.ndarray,
               n_samples: int,
               perm_subset: int,
               perm: list,
               parity: list,
               analysis_data: dict,
               iteration: int,
               load_weights: int,
               bosonic: bool,
               U: float,
               n_particles: int,
               dim_physical: int,
               n_layers: int,
               layer_size: int,
               epochs: int,
               batch_size: int,
               reg: float,
               GD: int = 0,
               normalize: bool = True,
               device: str = None,
               ):
    """
    Entraîne un réseau de neurones pour approximer la fonction d'onde,
    puis retourne deux callables :

        fitfunc(x_np)   → np.ndarray (batch,)   valeurs de ψ
        d2_fitfunc(x_t) → torch.Tensor (batch,) laplacien ∇²ψ

    Paramètres
    ----------
    x           : (N, dim) coordonnées des échantillons (numpy)
    psi         : (N, 1) valeurs de la fonction d'onde (numpy, réel)
    analysis_data : dict mutable — 'history' (loss) y est stocké après fit
    load_weights: 1 pour charger les poids sauvegardés, 0 pour entraîner
    device      : 'cuda', 'cpu' ou None (auto-détection)
    """

    # --- Device ---
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    dim = x.shape[1]

    # --- Modèle ---
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
        # --- Données ---
        x_t = torch.tensor(x, dtype=torch.float32, device=dev)
        y_t = torch.tensor(psi.ravel(), dtype=torch.float32, device=dev)

        dataset = TensorDataset(x_t, y_t)
        val_size = int(0.33 * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        # --- Optimiseur : SGD avec momentum (identique à la version TF) ---

        if GD == 0:
            optimizer = optim.Adam(model.parameters(), lr=0.2,betas=(0.9, 0.999), weight_decay=1e-5)
        else:
            optimizer = optim.SGD(model.parameters(),
                                lr=0.2,      
                                momentum=0.9,
                                weight_decay=1e-5,   
                                nesterov=False)
        # Décroissance du learning rate (équivalent à decay=1e-5 de Keras)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 1e-5)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-4)
        
        mse_loss = nn.MSELoss()
        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

        print("Entraînement en cours...")
        model.train()
        for epoch in range(epochs):
            # --- Epoch d'entraînement ---
            train_loss, train_mae, n_train = 0.0, 0.0, 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                #loss = mse_loss(pred, yb) + model.l2_loss()   #si on veut ajouter la régularisation à la main L2
                loss = mse_loss(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
                train_mae  += (pred - yb).abs().mean().item() * len(xb)
                n_train    += len(xb)
            #scheduler.step(train_loss/n_train)  # ajustement du learning rate selon la validation

            # --- Validation ---
            model.eval()
            val_loss, val_mae, n_val = 0.0, 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    loss = mse_loss(pred, yb)
                    val_loss += loss.item() * len(xb)
                    val_mae  += (pred - yb).abs().mean().item() * len(xb)
                    n_val    += len(xb)
            model.train()

            history['loss'].append(train_loss / n_train)
            history['val_loss'].append(val_loss / n_val)
            history['mae'].append(train_mae / n_train)
            history['val_mae'].append(val_mae / n_val)

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  "
                      f"loss={history['loss'][-1]:.4e}  "
                      f"val_loss={history['val_loss'][-1]:.4e}")

        analysis_data['history']  = history
        torch.save(model.state_dict(), ckpt_path)
        print(f"Poids sauvegardés dans {ckpt_path}")

    model.eval()

    def fitfunc(x_np: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Évalue ψ_θ sur un tableau numpy, retourne un tableau numpy."""
        results = []
        x_t = torch.tensor(x_np, dtype=torch.float32, device=dev)
        with torch.no_grad():
            for i in range(0, len(x_t), batch_size):
                results.append(model(x_t[i:i + batch_size]).cpu().numpy())
        out = np.concatenate(results, axis=0)
        return out.reshape(-1,1)


    def d2_fitfunc(x_input) -> torch.Tensor:
        if isinstance(x_input, np.ndarray):
            x_t = torch.tensor(x_input, dtype=torch.float32, device=dev)
        else:
            x_t = x_input.to(dev)
        
        x_t = x_t.detach().requires_grad_(True)

        psi_val = model(x_t)

        grad1 = torch.autograd.grad(
            outputs=psi_val,
            inputs=x_t,
            grad_outputs=torch.ones_like(psi_val),
            create_graph=True,
            retain_graph=True
        )[0]

        laplacian = torch.zeros(x_t.shape[0], device=dev)
        for i in range(dim):
            grad2_i = torch.autograd.grad(
                outputs=grad1[:, i],
                inputs=x_t,
                grad_outputs=torch.ones(x_t.shape[0], device=dev),
                retain_graph=(i < dim - 1),
                create_graph=False
            )[0]
            laplacian += grad2_i[:, i]
        laplacian=torch.clamp(laplacian, -1e5, 1e5)  # éviter les valeurs extrêmes
        
        max_ratio = 1e3
        psi_abs = psi_val.detach().abs() + 1e-10
        laplacian = torch.clamp(laplacian,
                            -max_ratio * psi_abs,
                            max_ratio * psi_abs)
        return laplacian.cpu().detach().reshape(-1, 1).numpy()

    return fitfunc, d2_fitfunc

##################################################################################
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from sample_distribution_Nd import sample_mixed
    import harmonic_oscillator_Nd as ho

    # Paramètres minimaux pour tester
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

    x0   = np.zeros(dim)
    step = np.full(dim, xmax)
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

    # Vérification rapide
    psi_pred = fitfunc(samples[:10])
    print("ψ prédit (10 premiers points) :", psi_pred)

    x_torch = torch.tensor(samples[:10], dtype=torch.float32)
    lap = d2_fitfunc(x_torch)
    print("∇²ψ (10 premiers points) :", lap)
    print("Test PyTorch OK ✓")