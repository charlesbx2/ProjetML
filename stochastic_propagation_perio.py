#!/usr/bin/env python
from matplotlib.pylab import norm
import numpy as np
import math
import random
import itertools
from sympy.combinatorics import Permutation
import sample_distribution_Nd
import sample_wavefunction_Nd
from params import functionaltype, periodic
if functionaltype == 1:
    import functional as nn_fit
else:
    import functional_neural_function_approximation_Nd as nn_fit
if periodic:
    import functionalperio as nn_fit
    from params import a, n_cells, V0, k_bloch
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle
import sys


# ---------------------------------------------------------------------------
# Permutation (cas non-périodique)
# ---------------------------------------------------------------------------

def permute(x_init, y_init, perm, parity, d, n_particles, bosonic):
    x_init  = x_init.reshape(x_init.shape[0], n_particles, d)
    num_perm = len(parity)
    x_perm  = np.copy(x_init)
    y_perm  = np.copy(y_init)
    for p in range(1, num_perm):
        permutation = list(perm[p])
        x_p = np.take(x_init, permutation, axis=1)
        x_perm = np.append(x_perm, x_p, axis=0)
        y_p = (1 if bosonic else (-1)**parity[p][0]) * y_init
        y_perm = np.append(y_perm, y_p)
    return x_perm.reshape(x_perm.shape[0], d * n_particles), y_perm


# ---------------------------------------------------------------------------
# Extraction de u_k depuis psi_k (cas périodique)
# ---------------------------------------------------------------------------

def extract_u_k(psi_samples, x_samples, k):
    """
    Extrait u_k = psi_k * e^{-ikx} et retourne la partie réelle.

    Inclut une correction de phase globale pour compenser la dérive
    accumulée par le propagateur Euler.
    """
    phase       = np.exp(-1j * k * x_samples[:, 0])
    u_k_complex = psi_samples.ravel() * phase

    # Correction de la dérive de phase globale
    mean_phase  = np.angle(u_k_complex.mean())
    u_k_complex = u_k_complex * np.exp(-1j * mean_phase)

    imag_ratio = (np.abs(u_k_complex.imag).mean() /
                  (np.abs(u_k_complex.real).mean() + 1e-10))
    if imag_ratio > 0.1:
        print(f"  WARNING: u_k partie imaginaire significative "
              f"(ratio={imag_ratio:.3f})")

    return u_k_complex.real


# ---------------------------------------------------------------------------
# Propagateur Euler en temps imaginaire
# ---------------------------------------------------------------------------

def propagate_samples(psi, d2psi, V, I, dt, x, m, hbar, n_particles, dim,
                      k=None):
    """
    psi_k(t+dt) = psi_k(t) - (i*dt/hbar) * H * psi_k(t)

    k est conservé pour compatibilité mais n'intervient plus dans
    le propagateur — la convergence vers l'état k est assurée par
    l'embedding de Fourier de BlochNet.
    """
    psi_x   = psi(x)
    d2psi_x = d2psi(x)
    return psi_x - (1j * dt / hbar) * (
        -(hbar**2) / (2 * m) * d2psi_x + (V(x) + I(x)) * psi_x
    )


# ---------------------------------------------------------------------------
# Méthode d'ajustement
# ---------------------------------------------------------------------------

def get_neural_fitting_method(bosonic, U, n_samples, perm_subset, n_particles,
                              dim_physical, n_layers, layer_size, epochs,
                              batch_size, reg):

    def fitting_method(x, y, perm, parity, analysis_data, iteration,
                       load_weights):
        return nn_fit.neural_fit(x, y, n_samples, perm_subset, perm, parity,
                                 analysis_data, iteration, load_weights,
                                 bosonic, U, n_particles, dim_physical,
                                 n_layers=n_layers, layer_size=layer_size,
                                 epochs=epochs, batch_size=batch_size,
                                 reg=reg)
    return fitting_method


def fit_samples(x, psi, fitting_method, perm, parity, iteration):
    psi          = psi.reshape(psi.shape[0], 1)
    load_weights = 1
    analysis_data = {}
    psi_input = np.real(psi) if not periodic else psi
    fitfunc, fitfunc_d2 = fitting_method(x, psi_input, perm, parity,
                                         analysis_data, iteration, load_weights)

    def fit_psi(x):   return fitfunc(x)[:, 0]
    def fit_d2psi(x): return fitfunc_d2(x)[:, 0]
    def fit_P(x):     return np.abs(fit_psi(x))**2

    return fit_psi, fit_d2psi, fit_P


# ---------------------------------------------------------------------------
# Construction des callables psi_k et ∇²psi_k depuis BlochNet
# ---------------------------------------------------------------------------

def make_bloch_callables(fitfunc, d2_fitfunc, k, a, h_fd=1e-4):
    """
    Depuis un réseau BlochNet entraîné sur u_k, construit :

        fit_psi(x)   = u_k(x % a) * e^{ikx}
        fit_d2psi(x) = e^{ikx} * (∇²u_k + 2ik∇u_k - k²u_k)
        fit_P(x)     = |u_k(x % a)|²

    ∇u_k et ∇²u_k sont obtenus depuis fitfunc et d2_fitfunc.
    Le repliement x % a est effectué à l'intérieur de fitfunc/d2_fitfunc
    (dans neural_fit), donc on n'a pas besoin de le refaire ici.
    """

    def fit_psi(x, _f=fitfunc):
        u     = _f(x)[:, 0]                         # u_k(x % a)
        phase = np.exp(1j * k * x[:, 0])
        return u * phase

    def fit_d2psi(x, _f=fitfunc, _d2=d2_fitfunc):
        u    = _f(x)[:, 0]                          # u_k
        d2u  = _d2(x)[:, 0]                         # ∇²u_k (depuis autograd)

        # ∇u_k par différences finies centrées sur u_k(x % a)
        x_ph = x.copy(); x_ph[:, 0] += h_fd
        x_mh = x.copy(); x_mh[:, 0] -= h_fd
        gradu = (_f(x_ph)[:, 0] - _f(x_mh)[:, 0]) / (2 * h_fd)

        phase = np.exp(1j * k * x[:, 0])
        # ∇²ψ_k = e^{ikx}(∇²u_k + 2ik∇u_k - k²u_k)
        return phase * (d2u + 2j * k * gradu - k**2 * u)

    def fit_P(x):
        return np.abs(fit_psi(x))**2

    return fit_psi, fit_d2psi, fit_P


# ---------------------------------------------------------------------------
# Propagation en temps
# ---------------------------------------------------------------------------

def propagate_in_time(iteration, eval_psi0, eval_V, eval_I, load_weights, U,
                      n_particles, dim_physical, nsamples, perm_subset, t, m,
                      hbar, xmax, n_x, step_size, x0, decorrelation_steps,
                      uniform_ratio, fitting_method, normalize, eta,
                      calculate_energy, bosonic):

    d      = dim_physical * n_particles
    perm   = list(itertools.permutations(range(n_particles)))
    parity = [[Permutation(list(p)).parity()] for p in perm]

    if iteration == 0:

        def eval_d2psi0(x):
            psi_ph = np.zeros(x.shape, dtype=complex)
            psi_mh = np.zeros(x.shape, dtype=complex)
            for i in range(d):
                x_ph = x.copy(); x_ph[:, i] += eta
                x_mh = x.copy(); x_mh[:, i] -= eta
                psi_ph[:, i] = eval_psi0(x_ph)
                psi_mh[:, i] = eval_psi0(x_mh)
            return (psi_ph.sum(axis=-1) + psi_mh.sum(axis=-1)
                    - 2 * d * eval_psi0(x)) / eta**2

        def P0(x):
            return np.real(np.conj(eval_psi0(x)) * eval_psi0(x))

        x_t        = np.zeros((nsamples, d, t.shape[0]))
        psi_t      = np.zeros((nsamples, t.shape[0]), dtype=complex)
        energies_t = np.zeros(t.shape[0])
        mse_t      = np.zeros(t.shape[0])
        d2psi_t    = np.zeros((nsamples, t.shape[0]), dtype=complex)

    else:
        filename   = sys.argv[1]
        results    = pickle.load(open(filename, "rb"))
        x_t        = results['x']
        x          = results['x'][:, :, iteration]
        psi_t      = results['psi_t']
        psi        = results['psi_t'][:, iteration]
        energies_t = results['energies_t'].real
        d2psi_t    = results.get('d2psi_t',
                                 np.zeros((nsamples, t.shape[0]), dtype=complex))
        mse_t      = results['mse_t'].real
        load_weights = 1
        eval_psi0_fit, eval_d2psi0_fit, P0 = fit_samples(
            x, psi, fitting_method, perm, parity, iteration)
        print(f'iteration: {iteration}, Energy: {energies_t[iteration]}, '
              f'Mse: {mse_t[iteration]}')

    x0_arr = np.zeros(d)
    step   = np.full(d, xmax)
    dt     = t[1] - t[0]

    eval_psi   = eval_psi0
    eval_d2psi = eval_d2psi0
    P          = P0
    start      = datetime.now()

    for i in range(iteration, t.shape[0]):
        print(f"i= {i} time= {datetime.now() - start}")

        samples      = sample_distribution_Nd.sample_mixed(
            P, x0_arr, step, xmax, nsamples, decorrelation_steps, uniform_ratio)
        x_t[:, :, i] = samples
        psi_t[:, i]  = eval_psi(samples)

        # --- Énergie ---
        if calculate_energy:
            def Hpsi(x):
                return (-(hbar**2) / (2*m) * eval_d2psi(x)
                        + eval_V(x) * eval_psi(x)
                        + eval_I(x) * eval_psi(x))

            energies_t[i], mse_t[i] = sample_wavefunction_Nd.vec_sample_energy(
                eval_psi, eval_d2psi, Hpsi, x0_arr, step, nsamples,
                decorrelation_steps, xmax)
            print(f"  Energy: {energies_t[i]:.6f}  Mse: {mse_t[i]:.4e}")
            print(f"  <V> = {np.mean(eval_V(samples)):.4f}")

        d2psi_vals    = eval_d2psi(samples)
        d2psi_t[:, i] = d2psi_vals[:nsamples]
        print(f"  ∇²ψ : min={d2psi_vals.min():.3e}, "
              f"max={d2psi_vals.max():.3e}, "
              f"nan={np.isnan(d2psi_vals).sum()}")

        # --- Propagation ---
        new_psi_t = propagate_samples(
            eval_psi, eval_d2psi, eval_V, eval_I, dt,
            samples, m, hbar, n_particles, dim_physical,
            k=k_bloch if periodic else None
        )

        # --- Protection NaN ---
        if not np.isfinite(np.abs(new_psi_t)).all():
            print(f"  WARNING: NaN au step {i}, on conserve le fit précédent")

        else:
            if periodic:
                # -----------------------------------------------------------
                # Cas périodique avec embedding de Fourier
                # -----------------------------------------------------------
                # 1. Extrait u_k depuis psi_k
                u_k = extract_u_k(new_psi_t, samples, k_bloch)

                # Diagnostic et plot
                print(f"  u_k : mean={u_k.mean():.4f}, "
                      f"std={u_k.std():.4f}, "
                      f"min={u_k.min():.4f}, max={u_k.max():.4f}")
                idx = np.argsort(samples[:, 0])
                plt.figure()
                plt.scatter(samples[idx, 0], u_k[idx], s=1)
                plt.plot(samples[idx, 0],
                         -np.cos(2*np.pi*samples[idx, 0]/a)*0.1 + u_k.mean(),
                         'r-', label='V(x) normalisé')
                plt.xlabel('x'); plt.ylabel('u_k(x)'); plt.legend()
                plt.savefig(f'uk_step_{i}.png'); plt.close()

                # Fixe le signe (u_k doit être majoritairement positif)
                if u_k.mean() < 0:
                    u_k = -u_k

                # 2. Normalisation L2
                norm_u = np.sqrt(np.mean(u_k**2))
                if norm_u < 1e-10:
                    print("  WARNING: norm(u_k) dégénérée")
                    norm_u = 1.0
                u_k = u_k / norm_u

                # 3. Pas d'augmentation — le réseau reçoit directement
                #    (x, u_k) avec repliement x % a effectué dans neural_fit
                x_train = samples          # (N, 1) dans [-L/2, L/2]
                y_train = u_k.reshape(-1, 1)

                # 4. Entraîne BlochNet sur u_k réel
                analysis_data = {}
                load_weights  = 0
                fitfunc, d2_fitfunc = fitting_method(
                    x_train, y_train, perm, parity, analysis_data, i,
                    load_weights
                )
                loss = analysis_data['history']['loss'][-1]
                print(f"  fit loss = {loss:.4e}")

                # 5. Reconstruit psi_k et ∇²psi_k depuis u_k
                eval_psi, eval_d2psi, P = make_bloch_callables(
                    fitfunc, d2_fitfunc, k_bloch, a)

            else:
                # -----------------------------------------------------------
                # Cas non-périodique (comportement original)
                # -----------------------------------------------------------
                psi = new_psi_t.real

                norm_val = np.sqrt(np.mean(np.abs(psi)**2))
                if normalize and norm_val > 1e-10:
                    psi = psi.reshape(psi.shape[0], 1) / norm_val

                perm_subset_actual = min(perm_subset, len(parity))
                subset = random.sample(np.arange(len(parity)).tolist(),
                                       perm_subset_actual)
                subset.sort()
                perm_i   = [perm[j] for j in subset]
                parity_i = [parity[j] for j in subset]

                x_aug, y_aug = permute(samples, psi, perm_i, parity_i,
                                       dim_physical, n_particles, bosonic)

                analysis_data = {}
                load_weights  = 0
                fitfunc, d2_fitfunc = fitting_method(
                    x_aug, y_aug, perm_i, parity_i, analysis_data, i,
                    load_weights
                )
                history = analysis_data['history']
                loss = (history.history['loss'][-1] if functionaltype == 0
                        else history['loss'][-1])

                def eval_psi(x, _f=fitfunc):
                    return _f(x)[:, 0]

                def eval_d2psi(x, _d2=d2_fitfunc):
                    return _d2(x)[:, 0]

                def P(x):
                    return np.abs(eval_psi(x))**2

        # --- Sauvegarde intermédiaire ---
        results = {
            'x':          x_t,
            't':          t,
            'psi_t':      psi_t,
            'energies_t': energies_t,
            'mse_t':      mse_t,
            'd2psi_t':    d2psi_t,
        }
        pickle.dump(results, open("intermediate_results.pkl", "wb"))

    print(f"total_time= {datetime.now() - start}")
    return x_t, psi_t, energies_t, mse_t, d2psi_t
