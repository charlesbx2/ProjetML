#!/usr/bin/env python
import numpy as np
import stochastic_propagation_Nd
import pickle
import sys

if __name__ == '__main__':
    sys.path.insert(0, '')
    from params import *

    # Neural network propagation
    t = np.linspace(0, tmax, n_t) * -1j
    nn_fitter = stochastic_propagation_Nd.get_neural_fitting_method(
        bosonic, U, nsamples, perm_subset, n_particles, dim_physical, n_layers,
        layer_size, epochs, batch_size, reg)
    x_t_nn, psi_t_nn, energies_t_nn, mse_t_nn, d2psi_t = stochastic_propagation_Nd.propagate_in_time(
        iteration, eval_psi0, eval_V, eval_I, load_weights, U, n_particles,
        dim_physical, nsamples, perm_subset, t, m, hbar, xmax, n_x, step_size,
        x0, decorrelation_steps, uniform_ratio, nn_fitter, normalize, eta,
        calculate_energy, bosonic)

    # File output
    results = {
        'x': x_t_nn,
        't': t,
        'psi_t': psi_t_nn,
        'energies_t': energies_t_nn,
        'mse_t': mse_t_nn,
        'd2psi_t': d2psi_t,
    }
    # Save results for plotting with name that changes with the parameters of the run
    filename = f"results_{n_particles}_{dim_physical}_{nsamples}_{functionaltype}_modifGauss.pkl"
    pickle.dump(results, open(filename, "wb"))
