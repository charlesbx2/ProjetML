#!/usr/bin/env python
import numpy as np
import math
import random
from math import pi
import matplotlib
import matplotlib.pyplot as plt
import itertools
from sympy.combinatorics import Permutation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Multiply, Reshape
from tensorflow.keras.optimizers import SGD
from datetime import datetime
import os
from os.path import join as pjoin
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
Use Gaussian function to create a boundary layer in order to suppress the fit function at the edges. 
"""


class GaussianLayer(tf.keras.layers.Layer):

    def __init__(self, n_particles, dim_physical, sigma_init):
        super(GaussianLayer, self).__init__()
        self.sigma = tf.Variable(initial_value=sigma_init, trainable=True)
        self.n_particles = tf.Variable(initial_value=n_particles,
                                       trainable=False)
        self.dim_physical = tf.Variable(initial_value=dim_physical,
                                        trainable=False)

    def build(self, input_shape):
        pass

    def call(self, input, bosonic):
        r = tf.reshape(input, [-1, self.n_particles, self.dim_physical])

        def gaussian(x):
            return K.exp(-(x * x) / (tf.constant(2.0) * self.sigma))

        def psi(x):
            return tf.reduce_prod(gaussian(x), axis=2, keepdims=False)

        return tf.reduce_prod(psi(r), axis=1, keepdims=False)


"""
Create neural network and train it to fit a multidimensional wavefunction. 
"""


def neural_fit(x,
               psi,
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
               n_layers,
               layer_size,
               epochs,
               batch_size,
               reg,
               GD=0,
               normalize=True):
    tf.keras.backend.clear_session()
    activation = 'gelu'
    dim = x.shape[1]

    def create_model():
        input = tf.keras.Input(shape=(dim, ), name='inputs')
        hidden = layers.Dense(layer_size,
                              input_dim=dim,
                              activation=activation,
                              kernel_initializer='uniform',
                              kernel_regularizer=regularizers.l2(reg))(input)
        for i in range(n_layers):
            hidden = layers.Dense(
                layer_size,
                activation=activation,
                kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(reg))(hidden)
        boundary = GaussianLayer(n_particles, dim_physical,
                                 sigma_init=2.0)(input, bosonic)
        output_initial = layers.Dense(units=1,
                                      activation='linear',
                                      name='outputs')(hidden)
        output = keras.layers.multiply([output_initial, boundary])

        model = keras.Model(inputs=[input], outputs=[output])

        opt = keras.optimizers.SGD(lr=0.2,
                                       decay=1e-5,
                                       momentum=0.9,
                                       nesterov=False)

        print("Compiling model...")
        model.compile(loss="mean_squared_error",
                      optimizer=opt,
                      metrics=['mae', 'accuracy'])
        return model

    model = create_model()

    dir = os.getcwd()
    dir_to_load_and_save = pjoin(
        dir, "checkpoints/my_checkpoint_{}".format(iteration))
    if load_weights == 1:
        model.load_weights(dir_to_load_and_save)
    else:
        print("Fitting...")
        history = model.fit(x,
                            psi,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.33)
        analysis_data['history'] = history
        model.save_weights(dir_to_load_and_save)

    def fitfunc(x, batch_size=128):
        return model.predict(x, batch_size)

    """
    Use tensorflow automatic gradient to calculate the Lagrangian of the fit function.  
    """

    @tf.function(experimental_relax_shapes=True)
    def d2_fitfunc(x):
        _x = tf.unstack(x, axis=1)
        _x_ = [tf.expand_dims(i, axis=1) for i in _x]
        _x2 = tf.transpose(tf.stack(_x_))[0]
        y = model(_x2)
        _grady = [tf.squeeze(tf.gradients(y, i)) for i in _x]
        grad2y = tf.stack(tf.gradients(_grady[0], _x_[0]))[0]
        for i in range(1, dim):
            grad2y = tf.concat(
                (grad2y, tf.stack(tf.gradients(_grady[i], _x_[i]))[0]), axis=1)
        grad2 = tf.reduce_sum(grad2y, axis=1, keepdims=True)
        return grad2

    return fitfunc, d2_fitfunc


if __name__ == '__main__':
    from sample_distribution_Nd import sample_mixed
    import stochastic_propagation_Nd
    import sample_wavefunction_Nd
    import harmonic_oscillator_Nd as ho

    # numerical parameters
    nsamples = 20000
    epochs = 2000
    batch_size = 10 * 128
    reg = 1e-8
    h = 1e-2
    n_layers = 2
    layer_size = 4 * 128
    bosonic = False
    perm_subset = 4
    load_weights = 0
    iteration = 0

    # physical parameters
    U = 0.0
    nu = 0.001
    hbar = 1.0
    m = 1.0
    omega = 1.0
    tmax = 1.0
    n_t = 100
    offset = [-1.0, 1.0, -1.0, 1.0]
    n_particles = 4
    dim_physical = 2

    t = -1j * np.linspace(0, tmax, n_t)
    dt = t[1] - t[0]

    def wavefunction(x):
        return ho.eigenfunction_samples(0,
                                        x,
                                        n_particles,
                                        dim_physical,
                                        offset,
                                        hbar,
                                        m,
                                        omega,
                                        bosonic=bosonic)

    def wavefunction_d2(x, h):
        psi_ph = np.zeros(x.shape, dtype=complex)
        psi_mh = np.zeros(x.shape, dtype=complex)
        for i in range(0, dim_physical * n_particles):
            x_ph = x.copy()
            x_mh = x.copy()
            x_ph[:, i] += h
            x_mh[:, i] -= h
            psi_ph[:, i] = wavefunction(x_ph)
            psi_mh[:, i] = wavefunction(x_mh)
        return (psi_ph.sum(axis=-1) + psi_mh.sum(axis=-1) - 2 *
                (dim_physical * n_particles) * wavefunction(x)) / (h**2)

    def P0(x):
        return np.real(np.conj(wavefunction(x) * wavefunction(x)))

    def eval_V(x):
        return ho.potential(x, m, omega)

    def eval_I(x):
        return ho.coulomb(x, U, nu)

    def Hpsi(x):
        return -(hbar**2) / (2 * m) * wavefunction_d2(
            x, h) + eval_V(x) * wavefunction(x) + eval_I(x) * wavefunction(x)

    x_max = 10.0
    step = np.full(dim_physical * n_particles, x_max)
    x0 = np.zeros(dim_physical * n_particles)
    decorrelation_steps = 10
    uniform_ratio = 0.1

    print("Generating samples...")
    start = datetime.now()
    samples = sample_mixed(P0, x0, step, x_max, nsamples, decorrelation_steps,
                           uniform_ratio)
    print("sampling time = ", datetime.now() - start)

    psi = wavefunction(samples).real
    average_value = np.max(np.abs(psi))
    psi = psi.reshape(psi.shape[0], 1) / average_value

    d = dim_physical * n_particles
    perm = list(itertools.permutations(range(n_particles)))
    parity = []
    for i in perm:
        parity.append([Permutation(i).parity()])

    subset = random.sample(np.arange(0, len(parity)).tolist(), perm_subset)
    subset.sort()
    perm = [perm[i] for i in subset]
    parity = [parity[i] for i in subset]
    x, y = permute(samples, psi, perm, parity, dim_physical, n_particles,
                   bosonic)

    # Fit a neural network to it:
    start = datetime.now()
    analysis_data = {}
    nn_fitfunc, d2_nn_fitfunc = neural_fit(samples, psi, nsamples, perm_subset,
                                           perm, parity, analysis_data,
                                           iteration, load_weights, bosonic, U,
                                           n_particles, dim_physical, n_layers,
                                           layer_size, epochs, batch_size, reg)
    print("training_duration = ", datetime.now() - start)
    start = datetime.now()

    def eval_psi(x):
        return nn_fitfunc(x)[:, 0]

    def eval_d2psi(x):
        f_d2 = (d2_nn_fitfunc(x).numpy())[:, 0]
        return f_d2

    def Hpsi(x):
        return -(hbar**2) / (2 * m) * eval_d2psi(x) + eval_V(x) * eval_psi(
            x) + eval_I(x) * eval_psi(x)

    energy, mse = sample_wavefunction_Nd.vec_sample_energy(
        eval_psi, eval_d2psi, Hpsi, x0, step, nsamples, decorrelation_steps,
        x_max)
    print("exact energy: ", energy_exact, "exact mse: ", mse_exact)
    print("NN energy: ", energy, "NN mse: ", mse)
    print("calculation_time_energy = ", datetime.now() - start)
