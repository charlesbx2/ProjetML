#!/usr/bin/env python
"""
run_bloch_vmc.py — VMC pour la relation de dispersion E(k)
V(x) = -V0 * cos(2π x / a)

u_k complexe = u_R + i*u_I représenté par deux BlochNet réels.

E[u] = ( ℏ²/2m * ∫[(∇uR)² + (∇uI)² + k²(uR²+uI²)]dx
         + ∫V(x)(uR²+uI²)dx ) / ∫(uR²+uI²)dx
"""

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from scipy.linalg import eigh
from functionalperio import BlochNet, BlochNetComplex
from params import a, V0, hbar, m

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
n_k_points  = 15
k_values    = np.linspace(-np.pi/a, np.pi/a, n_k_points)

n_harmonics = 20
n_layers    = 2
layer_size  = 2*64

n_steps_vmc = 10000
lr_vmc      = 1e-2
n_quad      = 500
print_every = 1000

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {dev}")
print(f"a={a}, V0={V0:.4f}, hbar={hbar}, m={m}\n")

# Grille de quadrature fixe sur [0, a]
x_np = np.linspace(0, a, n_quad, dtype=np.float32)
dx   = float(x_np[1] - x_np[0])
x_t  = torch.tensor(x_np.reshape(-1, 1), device=dev,
                    dtype=torch.float32, requires_grad=True)
V_t  = torch.tensor(
    (-V0 * np.cos(2 * np.pi * x_np / a)).astype(np.float32),
    device=dev
)

# ---------------------------------------------------------------------------
# Solution exacte
# ---------------------------------------------------------------------------

def exact_energy(k, V0, a, hbar, m, n_pw=30):
    G  = np.array([2*np.pi*n/a for n in range(-n_pw, n_pw+1)])
    N  = len(G)
    H  = np.diag((hbar**2/(2*m)) * (k+G)**2).astype(complex)
    G1 = 2*np.pi/a
    for i in range(N):
        for j in range(N):
            if abs(G[i]-G[j]-G1) < 1e-10 or abs(G[i]-G[j]+G1) < 1e-10:
                H[i,j] += -V0/2
    w, _ = eigh(H)
    return float(w[0].real)

print("Calcul des énergies exactes...")
E_exact_arr = np.array([exact_energy(k, V0, a, hbar, m) for k in k_values])
print(f"  E_exact : min={E_exact_arr.min():.4f}, max={E_exact_arr.max():.4f}\n")

# ---------------------------------------------------------------------------
# Loss variationnelle — u complexe = uR + i*uI
# ---------------------------------------------------------------------------

def variational_energy(model, k, n_samples=500):
    """
    Estimateur VMC classique inspiré de Mateos & Fraxanet.
    Sample des points selon |uR|² + |uI|², calcule E_loc et le gradient.
    """
    k_t = torch.tensor(float(k), dtype=torch.float32, device=dev)

    # 1. Évaluation sur grille pour estimer la densité
    with torch.no_grad():
        uR_g = model.net_real(x_t).squeeze()
        uI_g = model.net_imag(x_t).squeeze()
        density = uR_g**2 + uI_g**2
        density = density / density.sum()

    # 2. Sample des points selon |u|²
    idx = torch.multinomial(density, n_samples, replacement=True)
    x_s = x_t[idx].detach().requires_grad_(True)

    # 3. Calcul de E_loc = (H_k u) / u sur ces points
    uR_s = model.net_real(x_s).squeeze()
    uI_s = model.net_imag(x_s).squeeze()

    graduR_s = torch.autograd.grad(
        uR_s.sum(), x_s, create_graph=True, retain_graph=True)[0].squeeze()
    graduI_s = torch.autograd.grad(
        uI_s.sum(), x_s, create_graph=True, retain_graph=True)[0].squeeze()

    d2uR_s = torch.autograd.grad(
        graduR_s.sum(), x_s, create_graph=True, retain_graph=True)[0].squeeze()
    d2uI_s = torch.autograd.grad(
        graduI_s.sum(), x_s, create_graph=True, retain_graph=True)[0].squeeze()

    V_s = -V0 * torch.cos(2 * torch.pi * x_s.squeeze() / a)

    # E_loc = Re[ (H_k u) / u ] pour u complexe
    # H_k u = -1/2(d²uR + 2ik*duR - k²uR) + V*uR  (partie réelle)
    #       + -1/2(d²uI + 2ik*duI - k²uI) + V*uI  (partie imaginaire)
    mod2_s = (uR_s**2 + uI_s**2).detach()

    # Re[ u* H_k u ] / |u|²
    E_loc = (
        (-0.5 * (d2uR_s * uR_s + d2uI_s * uI_s)
         + k_t * (graduR_s * uI_s - graduI_s * uR_s)  # terme croisé ik
         + (0.5 * k_t**2 + V_s) * mod2_s)
        / mod2_s.clamp(min=1e-10)
    )

    E_mean = E_loc.mean()

    # 4. Loss = variance de E_loc (zéro à la solution exacte)
    loss = ((E_loc - E_mean.detach())**2).mean()

    return loss, float(E_mean.detach())


###########################################################################
print("=== TEST GRADIENT ===")
model_g = BlochNetComplex(a=a, n_harmonics=n_harmonics,
                          n_layers=n_layers, layer_size=layer_size).to(dev)

x_test = torch.tensor([[0.3]], device=dev, dtype=torch.float32, requires_grad=True)
uR_test = model_g.net_real(x_test).squeeze()
uI_test = model_g.net_imag(x_test).squeeze()

gR = torch.autograd.grad(uR_test, x_test, create_graph=False)[0]
gI = torch.autograd.grad(uI_test, x_test, create_graph=False)[0]
print(f"  ∂uR/∂x = {gR.item():.6f}")
print(f"  ∂uI/∂x = {gI.item():.6f}")
print("=== FIN TEST GRADIENT ===")

print(f"x_t.requires_grad = {x_t.requires_grad}")
###########################################################################
print("=== TEST DIAGNOSTIC ===")
k_test = np.pi / a

# Crée un modèle temporaire pour le test
model_test = BlochNetComplex(a=a, n_harmonics=n_harmonics,
                             n_layers=n_layers,
                             layer_size=layer_size).to(dev)

# Force uR ≈ cste, uI = 0
with torch.no_grad():
    for p in model_test.net_real.parameters():
        p.data.fill_(0.01)   # petit poids uniforme → sortie quasi-constante
    for p in model_test.net_imag.parameters():
        p.data.zero_()

_, E_test = variational_energy(model_test, k_test)
print(f"  u≈cste, k=π/a → E={E_test:.4f}  (attendu ≈ {k_test**2/2:.4f})")

# Force uR = cos(2πx/a), uI = 0  → test avec structure
_, E_test2 = variational_energy(model_test, 0.0)
print(f"  u≈cste, k=0   → E={E_test2:.4f}  (attendu ≈ 0 + <V>≈0)")

print("=== FIN TEST ===\n")

with torch.no_grad():
    for name, p in model_test.net_real.named_parameters():
        p.data.zero_()
    # Met le biais de la dernière couche à 1 pour avoir uR=1 partout
    model_test.net_real.net[-1].bias.data.fill_(1.0)
    for p in model_test.net_imag.parameters():
        p.data.zero_()

_, E_test = variational_energy(model_test, k_test)
print(f"  uR=1 exact, k=π/a → E={E_test:.4f}  (attendu={k_test**2/2:.4f})")
# ---------------------------------------------------------------------------
# VMC sweep
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# VMC sweep
# ---------------------------------------------------------------------------

print("=" * 55)
print("VMC — sweep sur la zone de Brillouin")
print("=" * 55)
 
E_vmc = []
 
for ik, k in enumerate(k_values):
    E_ex = E_exact_arr[ik]
    print(f"\nk={k/np.pi:.3f}π/a  ({ik+1}/{n_k_points})  "
          f"E_exact={E_ex:.4f}")
 
    # Initialisation aléatoire standard — PyTorch Kaiming par défaut
    model = BlochNetComplex(a=a, n_harmonics=n_harmonics,
                            n_layers=n_layers,
                            layer_size=layer_size).to(dev)
 
    _, E_pre = variational_energy(model, k)
    print(f"  Init aléatoire : E={E_pre:.4f}")
 
    # VMC — lr fixe, pas de scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr_vmc)
 
    best_E    = float('inf')
    best_state = None
 
    for step in range(1, n_steps_vmc + 1):
        optimizer.zero_grad()
        loss, E_val = variational_energy(model, k)
 
        if not np.isfinite(E_val):
            print(f"  Step {step} : E non finie, arrêt")
            break
 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
 
        if E_val < best_E:
            best_E     = E_val
            best_state = {kk: v.clone()
                          for kk, v in model.state_dict().items()}
 
        if step % print_every == 0:
            with torch.no_grad():
                uR_d = model.net_real(x_t).squeeze()
                uI_d = model.net_imag(x_t).squeeze()
            print(f"  Step {step:4d}  E={E_val:.4f}  best={best_E:.4f}  "
                  f"exact={E_ex:.4f}  err={abs(best_E-E_ex):.4e}  "
                  f"uR_std={uR_d.std():.3f}  uI_std={uI_d.std():.3f}")
 
    E_vmc.append(best_E)
    print(f"  → E_vmc={best_E:.6f}  E_exact={E_ex:.6f}  "
          f"err={abs(best_E-E_ex):.4e}")
 
E_vmc = np.array(E_vmc)

# ---------------------------------------------------------------------------
# Résumé
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("RÉSUMÉ")
print("=" * 55)
print(f"  MAE     = {np.mean(np.abs(E_vmc - E_exact_arr)):.4e}")
print(f"  Max err = {np.max(np.abs(E_vmc - E_exact_arr)):.4e}")

pickle.dump({'k': k_values, 'E_vmc': E_vmc, 'E_exact': E_exact_arr},
            open("dispersion.pkl", "wb"))
print("Résultats sauvegardés : dispersion.pkl")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

k_plot = k_values / (np.pi / a)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Relation de dispersion — V₀={V0:.2f}, a={a}', fontsize=13)

ax = axes[0]
ax.plot(k_plot, E_exact_arr, 'k-', lw=2, label='Exact (ondes planes)')
ax.plot(k_plot, E_vmc,       'ro', ms=5, label='BlochNet VMC')
ax.set_xlabel('k  [π/a]'); ax.set_ylabel('E(k)')
ax.set_title('Relation de dispersion'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
err = np.abs(E_vmc - E_exact_arr)
ax.semilogy(k_plot, err, 'b-o', ms=4)
ax.set_xlabel('k  [π/a]'); ax.set_ylabel('|E_vmc − E_exact|')
ax.set_title('Erreur absolue'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('dispersion.png', dpi=150)
print("Plot sauvegardé : dispersion.png")
plt.show()