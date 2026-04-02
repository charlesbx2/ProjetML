#!/usr/bin/env python
"""
Script de test autonome pour la régression de u_k sur une cellule unitaire.

- u_k est appris sur [0, a] uniquement
- L'énergie est calculée sur [0, a] par différences finies
- La périodicité est imposée par le repliement x % a

Usage :
    python test_regression_uk.py

Dépendances : torch, numpy, scipy, matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time

# ---------------------------------------------------------------------------
# Paramètres physiques
# ---------------------------------------------------------------------------
a    = 1.0
V0   = 2 * np.pi**2
hbar = 1.0
m    = 1.0
k    = 0.3 * np.pi / a

n_samples = 2000

# ---------------------------------------------------------------------------
# Calcul exact de u_k par diagonalisation en ondes planes
# ---------------------------------------------------------------------------

def compute_exact_uk(k, V0, a, hbar, m, n_pw=20):
    G_list = np.array([2 * np.pi * n / a for n in range(-n_pw, n_pw + 1)])
    N = len(G_list)
    H = np.zeros((N, N), dtype=complex)
    for i, G in enumerate(G_list):
        H[i, i] = (hbar**2 / (2 * m)) * (k + G)**2
    G1 = 2 * np.pi / a
    for i in range(N):
        for j in range(N):
            dG = G_list[i] - G_list[j]
            if np.abs(dG - G1) < 1e-10 or np.abs(dG + G1) < 1e-10:
                H[i, j] += -V0 / 2
    eigenvalues, eigenvectors = eigh(H)
    return eigenvalues[0], eigenvectors[:, 0], G_list


def eval_uk_exact(x, c_vec, G_list):
    u = np.zeros(len(x), dtype=complex)
    for c, G in zip(c_vec, G_list):
        u += c * np.exp(1j * G * x)
    return u.real


# ---------------------------------------------------------------------------
# Calcul de l'énergie sur [0, a]
# ---------------------------------------------------------------------------

def compute_energy_on_cell(fitfunc, dev, k, a, hbar, m, V0,
                            n_points=2000, h_fd=1e-5):
    """
    Énergie <H> calculée par quadrature sur [0, a].

    psi_k(x) = fitfunc(x) * e^{ikx}   avec x ∈ [0, a]
    ∇²psi_k calculé par différences finies.
    Pondération par |psi_k|².
    """
    x_quad = np.linspace(0, a, n_points)

    def psi_k(x_arr):
        xt = torch.tensor(x_arr.reshape(-1, 1), dtype=torch.float32, device=dev)
        with torch.no_grad():
            u = fitfunc(xt).cpu().numpy().ravel()
        return u * np.exp(1j * k * x_arr)

    psi_0  = psi_k(x_quad)
    psi_ph = psi_k(x_quad + h_fd)
    psi_mh = psi_k(x_quad - h_fd)
    d2psi  = (psi_ph + psi_mh - 2 * psi_0) / h_fd**2

    V_arr = -V0 * np.cos(2 * np.pi * x_quad / a)

    with np.errstate(divide='ignore', invalid='ignore'):
        E_local = np.where(
            np.abs(psi_0) > 1e-10,
            (-hbar**2 / (2*m) * d2psi / psi_0 + V_arr).real,
            0.0
        )

    weights = np.abs(psi_0)**2
    norm_w  = np.sum(weights)
    if norm_w < 1e-10:
        return np.nan

    return float((np.sum(E_local * weights) / norm_w).real)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class WavefunctionNet(nn.Module):
    def __init__(self, n_layers, layer_size, activation='gelu'):
        super().__init__()
        def act():
            return nn.GELU() if activation == 'gelu' else nn.Tanh()

        layers = [nn.Linear(1, layer_size), act()]
        for _ in range(n_layers):
            layers += [nn.Linear(layer_size, layer_size), act()]
        layers.append(nn.Linear(layer_size, 1))
        self.net = nn.Sequential(*layers)

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Entraînement + évaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(config, x_train, u_train, x_test, u_test_exact,
                       E_exact, verbose=True):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = WavefunctionNet(
        config['n_layers'],
        config['layer_size'],
        config.get('activation', 'gelu')
    ).to(dev)

    x_t = torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32, device=dev)
    y_t = torch.tensor(u_train,                dtype=torch.float32, device=dev)

    dataset  = TensorDataset(x_t, y_t)
    val_size = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False)

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                               weight_decay=config.get('weight_decay', 0))
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                              momentum=0.9, nesterov=False,
                              weight_decay=config.get('weight_decay', 0))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    mse_loss = nn.MSELoss()
    history  = {'train': [], 'val': []}
    t0 = time.time()

    for epoch in range(config['epochs']):
        model.train()
        tl, n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(xb)
            n  += len(xb)

        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                vl += mse_loss(model(xb), yb).item() * len(xb)
                nv += len(xb)

        tl /= n;  vl /= nv
        history['train'].append(tl)
        history['val'].append(vl)
        scheduler.step(vl)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}/{config['epochs']}  "
                  f"train={tl:.4e}  val={vl:.4e}")

    duration = time.time() - t0
    model.eval()

    # Prédiction sur grille de test
    xt = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32, device=dev)
    with torch.no_grad():
        u_pred = model(xt).cpu().numpy()

    if np.corrcoef(u_pred, u_test_exact)[0, 1] < 0:
        u_pred = -u_pred

    mse_test = float(np.mean((u_pred - u_test_exact)**2))
    max_err  = float(np.max(np.abs(u_pred - u_test_exact)))

    # Correction de signe pour le calcul d'énergie
    sign = 1.0
    xt2  = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32, device=dev)
    with torch.no_grad():
        u_check = model(xt2).cpu().numpy()
    if np.corrcoef(u_check, u_test_exact)[0, 1] < 0:
        sign = -1.0

    def fitfunc_signed(x_t):
        return sign * model(x_t)

    E_nn = compute_energy_on_cell(fitfunc_signed, dev, k, a, hbar, m, V0)

    return {
        'mse_test': mse_test,
        'max_err':  max_err,
        'E_nn':     E_nn,
        'E_exact':  E_exact,
        'E_error':  abs(E_nn - E_exact) if not np.isnan(E_nn) else np.inf,
        'duration': duration,
        'history':  history,
        'u_pred':   u_pred,
    }


# ---------------------------------------------------------------------------
# Génération des données
# ---------------------------------------------------------------------------

print("=" * 60)
print("Calcul de u_k exact...")
E_exact, c_vec, G_list = compute_exact_uk(k, V0, a, hbar, m, n_pw=20)
print(f"  E(k={k/np.pi:.2f}π/a) = {E_exact:.6f}")
print(f"  k²/2                  = {k**2/2:.6f}")
print(f"  Correction potentiel  = {E_exact - k**2/2:.6f}")

np.random.seed(42)
x_cell = np.random.uniform(0, a, n_samples)
u_cell = eval_uk_exact(x_cell, c_vec, G_list)
norm   = np.sqrt(np.mean(u_cell**2))
u_cell = u_cell / norm
if u_cell.mean() < 0:
    u_cell = -u_cell

x_test       = np.linspace(0, a, 500)
u_test_exact = eval_uk_exact(x_test, c_vec, G_list) / norm
if u_test_exact.mean() < 0:
    u_test_exact = -u_test_exact

print(f"\nDataset : {n_samples} points dans [0, a]")
print(f"  u_k : mean={u_cell.mean():.4f}, std={u_cell.std():.4f}, "
      f"min={u_cell.min():.4f}, max={u_cell.max():.4f}")

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
configs = [
    {'name': 'SGD lr=0.2  2x64',
     'n_layers': 2, 'layer_size': 64,  'lr': 0.2,
     'optimizer': 'sgd',  'epochs': 500, 'batch_size': 128,
     'weight_decay': 1e-5, 'activation': 'gelu'},

    {'name': 'Adam lr=1e-3 2x64',
     'n_layers': 2, 'layer_size': 64,  'lr': 1e-3,
     'optimizer': 'adam', 'epochs': 500, 'batch_size': 128,
     'weight_decay': 1e-5, 'activation': 'gelu'},

    {'name': 'Adam lr=1e-3 4x256',
     'n_layers': 4, 'layer_size': 256, 'lr': 1e-3,
     'optimizer': 'adam', 'epochs': 500, 'batch_size': 256,
     'weight_decay': 1e-5, 'activation': 'gelu'},

    {'name': 'Adam lr=1e-3 2x128 tanh',
     'n_layers': 2, 'layer_size': 128, 'lr': 1e-3,
     'optimizer': 'adam', 'epochs': 500, 'batch_size': 128,
     'weight_decay': 1e-5, 'activation': 'tanh'},

    {'name': 'Adam lr=3e-3 3x128',
     'n_layers': 3, 'layer_size': 128, 'lr': 3e-3,
     'optimizer': 'adam', 'epochs': 500, 'batch_size': 128,
     'weight_decay': 1e-4, 'activation': 'gelu'},

    {'name': 'Adam lr=1e-2 2x64',
     'n_layers': 2, 'layer_size': 64,  'lr': 1e-2,
     'optimizer': 'adam', 'epochs': 500, 'batch_size': 64,
     'weight_decay': 1e-5, 'activation': 'gelu'},

    {'name': 'Adam lr=1e-3 2x64 tanh',
     'n_layers': 2, 'layer_size': 64,  'lr': 1e-3,
     'optimizer': 'adam', 'epochs': 500, 'batch_size': 128,
     'weight_decay': 1e-5, 'activation': 'tanh'},
]

print("\n" + "=" * 60)
print("GRID SEARCH — régression sur [0, a]")
print("=" * 60)

results = {}
for cfg in configs:
    name = cfg['name']
    print(f"\n--- {name} ---")
    res = train_and_evaluate(cfg, x_cell, u_cell,
                             x_test, u_test_exact, E_exact, verbose=True)
    results[name] = res
    print(f"  MSE[0,a]   = {res['mse_test']:.4e}")
    print(f"  Max erreur = {res['max_err']:.4e}")
    print(f"  E_nn       = {res['E_nn']:.6f}  "
          f"(exact={E_exact:.6f}, err={res['E_error']:.4e})")
    print(f"  Durée      = {res['duration']:.1f}s")

# ---------------------------------------------------------------------------
# Résumé
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RÉSUMÉ")
print("=" * 60)
print(f"{'Config':<25} {'MSE[0,a]':>12} {'|E_nn-E_exact|':>16} {'Durée':>8}")
print("-" * 65)
for name, res in results.items():
    print(f"{name:<25} {res['mse_test']:>12.4e} "
          f"{res['E_error']:>16.4e} {res['duration']:>7.1f}s")

best_name = min(results, key=lambda n: results[n]['E_error'])
best      = results[best_name]
print(f"\nMeilleure config : {best_name}")
print(f"  E_nn = {best['E_nn']:.6f}  (exact = {E_exact:.6f})")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'Régression u_k sur [0,a] — k={k/np.pi:.2f}π/a, V0=2π²={V0:.1f}',
    fontsize=13
)

# 1. u_k exact + samples
ax = axes[0, 0]
ax.scatter(x_cell, u_cell, s=2, alpha=0.3, c='gray', label='samples')
ax.plot(x_test, u_test_exact, 'k-', lw=2, label='u_k exact')
V_norm = (-np.cos(2*np.pi*x_test/a) * u_test_exact.std()*0.5
          + u_test_exact.mean())
ax.plot(x_test, V_norm, 'r--', alpha=0.6, label='V(x) normalisé')
ax.set_title('u_k exact et samples sur [0, a]')
ax.set_xlabel('x');  ax.legend(fontsize=8)

# 2. Prédictions
ax = axes[0, 1]
ax.plot(x_test, u_test_exact, 'k-', lw=2.5, label='Exact', zorder=5)
for (name, res), c in zip(results.items(), colors):
    ax.plot(x_test, res['u_pred'], '--', color=c, alpha=0.8,
            label=f"{name} (Eerr={res['E_error']:.2f})")
ax.set_title('u_k : exact vs réseaux')
ax.set_xlabel('x');  ax.legend(fontsize=7)

# 3. Loss
ax = axes[1, 0]
for (name, res), c in zip(results.items(), colors):
    ep = range(1, len(res['history']['train']) + 1)
    ax.semilogy(ep, res['history']['train'], '-',  color=c, alpha=0.9,
                label=name)
    ax.semilogy(ep, res['history']['val'],   '--', color=c, alpha=0.5)
ax.set_title('Loss train (solide) / val (tiret)')
ax.set_xlabel('Epoch');  ax.set_ylabel('MSE');  ax.legend(fontsize=7)

# 4. Énergie estimée
ax = axes[1, 1]
names_s  = list(results.keys())
E_values = [res['E_nn'] for res in results.values()]
ax.bar(range(len(names_s)), E_values,
       color=colors[:len(names_s)], alpha=0.7)
ax.axhline(E_exact,   color='k', lw=2,   linestyle='--',
           label=f'E exact = {E_exact:.3f}')
ax.axhline(k**2/2,   color='r', lw=1.5, linestyle=':',
           label=f'k²/2 = {k**2/2:.3f}')
ax.set_xticks(range(len(names_s)))
ax.set_xticklabels([n[:18] for n in names_s],
                   rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Énergie')
ax.set_title('Énergie estimée vs exacte')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('test_regression_uk_cell.png', dpi=150)
print("\nPlot sauvegardé : test_regression_uk_cell.png")
plt.show()
