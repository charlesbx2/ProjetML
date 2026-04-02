### Physical parameters ###
U = 0.0  # Coulomb interaction
nu = 0.001  # Offset of Coulomb interaction to avoid divergence
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Mass for harmonic oscillator potential
omega = 1.0  # Frequency for harmonic oscillator potential
tmax = 6 # Maximal time of imaginary time propagation
xmax = 10.0  # Cutoff for generating samples in space
offset = [-1.0, 1.0]  # Offset of initial non-interacting wavefunction
normalize = True  # Normalize maximal value of wavefunction to 1 after every step?
bosonic = False  # Use bosonic symmetry? Otherwise use fermionic
calculate_energy = True  # Calculate energy at every step of the imaginary time propagation?
n_particles = 2  # Number of particles
dim_physical = 2  # Number of spatial dimensions

### Numerical parameters ###
iteration = 0  # Initial step in the imaginary time propagation; the default is to start with 0
n_t = 30 # Imaginary time propagation steps
n_x = 100  # Size of grid in space
nsamples = 2000  # Number of samples used to represent the wavefunction
decorrelation_steps = 5  # Decorrelation of samples in Monte Carlo
uniform_ratio = 0.2  # Ratio of uniformly distributes samples
step_size = xmax
x0 = 0.0
eta = 1e-2  # Discretization for finite difference
perm_subset = 1  # !!!!!!!!!!!!!!!!!!! Number of permutations to be considered at every iteration step; maximum value is n_particles!
load_weights = 0  # Load neural network weights from previous fit

### Modifications for neural network implementation ###

functionaltype = 1 # 0 if original version, 1 if pytorch

periodic = True # Use periodic potential instead of harmonic oscillator potential


### Neural network parameters ###
n_layers = 2
layer_size = 1 * 64
epochs = 500
batch_size = 128
reg = 1e-8

### Set up initial condition and potential evaluation: ###
import harmonic_oscillator_Nd as ho
import numpy as np
"""
Define functions to pass on to the propagation algorithm. 
"""


def eval_psi0(x):
    return ho.eigenfunction_samples(0, x, n_particles, dim_physical, offset,
                                    hbar, m, omega, bosonic)


def eval_V(x):
    return ho.potential(x, m, omega)

#potentiel périodique

def eval_V_periodic(x):
    return ho.potential_periodic(x, V0, a)

if periodic:
    n_particles  = 1
    dim_physical = 1
    U            = 0.0
    a            = 1.0        # paramètre de maille
    n_cells      = 1        # nombre de cellules dans la supercell
    L            = n_cells * a  # taille de la supercell
    V0           = 1*(2*np.pi**2)**0       # profondeur du potentiel
    k_bloch      = 0.5*np.pi / a  # vecteur de Bloch (bord de zone : k=π/a)
                            # ou k=0 pour l'état fondamental

    xmax         = L / 2      # échantillonnage sur [-L/2, L/2]
    offset       = [0.0]      # une seule particule, pas d'offset
    n_harmonics   = 8          # nombre d'harmoniques à considérer dans la fonction d'onde de Bloch

    def eval_V(x):
        return ho.potential_periodic(x, V0, a)
    
    def eval_psi0(x):
        return ho.bloch_initial_state(x, k_bloch, V0, a, hbar, m)



def eval_I(x):
    return ho.coulomb(x, U, nu)


def eval_exact_energy(x):
    return ho.energy(0, x, U, nu, hbar, omega)
