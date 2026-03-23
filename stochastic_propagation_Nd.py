#!/usr/bin/env python
import numpy as np
import math
import random
import itertools
from sympy.combinatorics import Permutation
import sample_distribution_Nd
import sample_wavefunction_Nd
from params import functionaltype
if functionaltype==1:
    import functional as nn_fit
else:
    import functional_neural_function_approximation_Nd as nn_fit
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle
import sys
"""
Given the sample points x and the values of the wavcefunction psi, generate permutations according ot the symmetry of the wavefunction. 
"""


def permute(x_init, y_init, perm, parity, d, n_particles, bosonic):
    x_init = x_init.reshape(x_init.shape[0], n_particles, d)
    num_perm = len(parity)
    x_perm = np.copy(x_init)
    y_perm = np.copy(y_init)
    for p in range(1, num_perm):
        permutation = list(perm[p])
        x_p = np.take(x_init, permutation, axis=1)
        x_perm = np.append(x_perm, x_p, axis=0)
        if (bosonic):
            y_p = (1)**(parity[p][0]) * y_init
        else:
            y_p = (-1)**(parity[p][0]) * y_init
        y_perm = np.append(y_perm, y_p)
    return x_perm.reshape(x_perm.shape[0], d * n_particles), y_perm


"""
Given functions for evaluating the wavefunction psi, its laplacian d2psi and the
potential V at time t, calculate the wavefunction at each of the sample points x
for time t + dt using the Euler method.
"""


def propagate_samples(psi, d2psi, V, I, dt, x, m, hbar, n_particles, dim):
    psi_x = psi(x)
    d2psi_x = d2psi(x)
    return psi_x - (1j * dt / hbar) * (-(hbar**2) / (2 * m) * d2psi_x +
                                       (V(x) + I(x)) * psi_x)


"""
Obtain a fitting method based on neural networks.
"""


def get_neural_fitting_method(bosonic, U, n_samples, perm_subset, n_particles,
                              dim_physical, n_layers, layer_size, epochs,
                              batch_size, reg):

    def fitting_method(x, y, perm, parity, analysis_data, iteration,
                       load_weights):
        return nn_fit.neural_fit(x,
                                 y,
                                 n_samples,
                                 perm_subset,
                                 perm,
                                 parity,
                                 analysis_data,
                                 iteration,
                                 load_weights,
                                 bosonic,
                                 U,
                                 n_particles,
                                 dim_physical,
                                 n_layers=n_layers,
                                 layer_size=layer_size,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 reg=reg)

    return fitting_method


"""
Obtain a fit function, its Lagrangian and the probability distribution. 
"""


def fit_samples(x, psi, fitting_method, perm, parity, iteration):
    psi = psi.reshape(psi.shape[0], 1)
    load_weights = 1
    analysis_data = {}
    fitfunc, fitfunc_d2 = fitting_method(x, np.real(psi), perm, parity,
                                         analysis_data, iteration,
                                         load_weights)

    def fit_psi(x):
        f = fitfunc(x)[:, 0]
        return f

    def fit_d2psi(x):
        f_d2 = fitfunc_d2(x).numpy()[:, 0]
        return f_d2
    
    if functionaltype == 1:
            def fit_d2psi(x):
                f_d2 = fitfunc_d2(x)[:, 0]
                return f_d2

    def fit_P(x):
        return np.abs(fit_psi(x)**2)

    return fit_psi, fit_d2psi, fit_P


"""
Perform time propagation.
"""


def propagate_in_time(iteration, eval_psi0, eval_V, eval_I, load_weights, U,
                      n_particles, dim_physical, nsamples, perm_subset, t, m,
                      hbar, xmax, n_x, step_size, x0, decorrelation_steps,
                      uniform_ratio, fitting_method, normalize, eta,
                      calculate_energy, bosonic):
    d = dim_physical * n_particles
    perm = list(itertools.permutations(range(n_particles)))
    parity = []
    for i in perm:
        parity.append([Permutation(i).parity()])

    if iteration == 0:

        def eval_d2psi0(x):
            psi_ph = np.zeros(x.shape, dtype=complex)
            psi_mh = np.zeros(x.shape, dtype=complex)
            for i in range(0, dim_physical * n_particles):
                x_ph = x.copy()
                x_mh = x.copy()
                x_ph[:, i] += eta
                x_mh[:, i] -= eta
                psi_ph[:, i] = eval_psi(x_ph)
                psi_mh[:, i] = eval_psi(x_mh)
            return (psi_ph.sum(axis=-1) + psi_mh.sum(axis=-1) - 2 *
                    (dim_physical * n_particles) * eval_psi(x)) / (eta**2)

        def P0(x):
            return np.real(np.conj(eval_psi0(x)) * eval_psi0(x))

        x_t = np.zeros((nsamples, n_particles * dim_physical, t.shape[0]))
        psi_t = np.zeros((nsamples, t.shape[0]), dtype=complex)
        energies_t = np.zeros(t.shape[0])
        mse_t = np.zeros(t.shape[0])
        d2psi_t = np.zeros((nsamples, t.shape[0]))

    else:
        filename = sys.argv[1]
        results = pickle.load(open(filename, "rb"))
        x_t = results['x']
        x = results['x'][:, :, iteration]
        psi_t = results['psi_t']
        psi = results['psi_t'][:, iteration]
        energies_t = results['energies_t'].real
        d2psi_t = results.get('d2psi_t', np.zeros((nsamples, t.shape[0])))
        mse_t = results['mse_t'].real
        load_weights = 1
        eval_psi0, eval_d2psi0, P0 = fit_samples(x, psi, fitting_method, perm,
                                                 parity, iteration)
        print('iteration: ', iteration, ', Energy: ', energies_t[iteration],
              ', Mse: ', mse_t[iteration])

    x0_arr = np.zeros(n_particles * dim_physical)
    step = np.full(n_particles * dim_physical, xmax)
    dt = t[1] - t[0]

    eval_psi = eval_psi0
    eval_d2psi = eval_d2psi0
    
    P = P0
    start = datetime.now()
    for i in range(iteration, t.shape[0]):
        print("i=", i, "time=", datetime.now() - start)
        samples = sample_distribution_Nd.sample_mixed(P, x0_arr, step, xmax,
                                                      nsamples,
                                                      decorrelation_steps,
                                                      uniform_ratio)
        x_t[:, :, i] = samples
        psi_t[:, i] = eval_psi(samples)

        if (calculate_energy):

            def Hpsi(x):
                return -(hbar**2) / (2 * m) * eval_d2psi(x) + eval_V(
                    x) * eval_psi(x) + eval_I(x) * eval_psi(x)

            energies_t[i], mse_t[i] = sample_wavefunction_Nd.vec_sample_energy(
                eval_psi, eval_d2psi, Hpsi, x0_arr, step, nsamples,
                decorrelation_steps, xmax)
            print("Energy: ", energies_t[i], "Mse: ", mse_t[i])

        d2psi_vals = eval_d2psi(samples)
        d2psi_t[:, i] = d2psi_vals[:nsamples]  # garde exactement nsamples valeurs
        print(f"  ∇²ψ : min={d2psi_vals.min():.3e}, "
        f"max={d2psi_vals.max():.3e}, "
        f"nan={np.isnan(d2psi_vals).sum()}")

        new_psi_t = propagate_samples(eval_psi, eval_d2psi, eval_V, eval_I, dt,
                                      samples, m, hbar, n_particles,
                                      dim_physical)
        psi = new_psi_t.real
        average_value = np.max(np.abs(psi))
        if normalize:
            average_value = np.max(np.abs(psi))
            psi = psi.reshape(psi.shape[0], 1) / average_value

        subset = random.sample(np.arange(0, len(parity)).tolist(), perm_subset)
        subset.sort()
        perm = [perm[i] for i in subset]
        parity = [parity[i] for i in subset]
        x, y = permute(samples, psi, perm, parity, dim_physical, n_particles,
                       bosonic)

        analysis_data = {}
        load_weights = 0
        fitfunc, d2_fitfunc = fitting_method(x, y, perm, parity, analysis_data,
                                             i, load_weights)
        history = analysis_data['history']
        if functionaltype == 0:
            loss = history.history['loss'][-1]
        else : 
            loss = history['loss'][-1]
        

        def fit_psi(x):
            f = fitfunc(x)[:, 0]
            return f

        def fit_d2psi(x):
            f_d2 = (d2_fitfunc(x).numpy())[:, 0]
            return f_d2
        
        if functionaltype == 1:
            def fit_d2psi(x):
                f_d2 = (d2_fitfunc(x))[:, 0]
                return f_d2

        def fit_P(x):
            return np.abs(fit_psi(x)**2)

        eval_psi = fit_psi
        eval_d2psi = fit_d2psi
        P = fit_P
        """
	Save intermediate output. 
	"""
        results = {
            'x': x_t,
            't': t,
            'psi_t': psi_t,
            'energies_t': energies_t,
            'mse_t': mse_t,
            'd2psi_t': d2psi_t,
        }
        pickle.dump(results, open("intermediate_results.pkl", "wb"))

    print("total_time=", datetime.now() - start)
    return x_t, psi_t, energies_t, mse_t, d2psi_t


if __name__ == '__main__':
    import harmonic_oscillator_Nd as ho
    import matplotlib.pyplot as plt
    U = 0.0
    nu = 0.01
    hbar = 1.0
    m = 1.0
    omega = 1.0
    tmax = 1.0
    xmax = 5.0
    offset = [0.0, 0.0]
    n_t = 100
    nsamples = 5000
    bosonic = False
    eta = 1e-2
    n_particles = 2
    dim_physical = 2

    def wavefunction(x):
        return ho.eigenfunction_samples(0, x, n_particles, dim_physical,
                                        offset, hbar, m, omega, bosonic)

    def P0(x):
        return np.real(np.conj(wavefunction(x)) * wavefunction(x))

    x0 = np.zeros(n_particles * dim_physical)
    step = np.full(n_particles * dim_physical, xmax)
    decorrelation_steps = 10
    uniform_ratio = 0.2
    samples = sample_distribution_Nd.sample_mixed(P0, x0, step, xmax, nsamples,
                                                  decorrelation_steps,
                                                  uniform_ratio)

    t = -1j * np.linspace(0, tmax, n_t)
    dt = t[1] - t[0]
    """
    Stochastic propagation with exact wavefunction. 
    """

    def eval_psi(x):
        return ho.eigenfunction_samples(0, x, n_particles, dim_physical,
                                        offset, hbar, m, omega, bosonic)

    def eval_d2psi(x, eta=1e-5):
        psi_ph = np.zeros(x.shape, dtype=complex)
        psi_mh = np.zeros(x.shape, dtype=complex)
        for i in range(0, dim_physical * n_particles):
            x_ph = x.copy()
            x_mh = x.copy()
            x_ph[:, i] += eta
            x_mh[:, i] -= eta
            psi_ph[:, i] = wavefunction(x_ph)
            psi_mh[:, i] = wavefunction(x_mh)
        return (psi_ph.sum(axis=-1) + psi_mh.sum(axis=-1) - 2 *
                (dim_physical * n_particles) * wavefunction(x)) / (eta**2)

    def eval_V(x):
        return ho.potential(x, m, omega)

    def eval_I(x):
        return ho.coulomb(x, U, nu)

    psi0 = eval_psi(samples)
    psi_t = propagate_samples(eval_psi, eval_d2psi, eval_V, eval_I, dt,
                              samples, m, hbar, n_particles, dim_physical)
