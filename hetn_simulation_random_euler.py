#
# This script simulates a heterogeneous Nagumo SPDE, where
# the localised noise is added multiplicatively. It uses
# a finite difference, with uniform discretization, scheme
# to discretize the space, and it solves the resulting SDE
# with the Euler method.
#
# The simulation corresponds to the evolution of E[U(t,x)], where
# E is the expectation and U is the solution of the SPDE.
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.sparse import diags
from matplotlib.ticker import MaxNLocator

# Physical parameters
gamma = 0.01
mu = 0.1
x0 = -5
delta = 0.1

# Numerical parameters
L = 10.0
N = 500
h = 2 * L / N
steps = 1000
t_steps = 1000 # Time steps per step. This is added for memory efficiency.
dt = h**2 / 3
T = dt * steps * t_steps
paths = 500
seed = 42

# Set initial condition
def U0(x):
    return np.tanh((x + 3) / np.sqrt(2))

# Define indicator function
def ind(x):
    return np.logical_and(x >= x0 - delta, x <= x0 + delta).astype(float)

# Solve IVP using finite difference and Euler-Maruyama scheme
def solve_ivp(u, Ah, indicator):
    Z = np.random.normal(0.0, 1.0, [N + 1, t_steps, paths])
    # Euler-Maruyama step
    for i in range(t_steps):
        if paths > 1: # Renormalize the normal distribution
            Z_i = Z[:, i, :] # Storage for efficiency
            Z[:, i, :] = (Z_i - np.mean(Z_i)) / np.std(Z_i)
        for w in range(paths):
            u_w = u[:, w]; Z_iw = Z[:, i, w] # Storage for efficiency
            u[:, w] = u_w + dt * (Ah @ u_w - u_w ** 3 + u_w + gamma) \
                             + mu * u_w * indicator * np.power(dt, 0.5) * Z_iw
    return u


def main():
    # Data structures
    np.random.seed(42)
    main_diag = -2.0 * np.ones(N + 1)
    below_diag = 1.0 * np.ones(N)
    below_diag[-1] = 2.0
    upper_diag = np.copy(below_diag)
    upper_diag[0] = 2.0
    Ah = diags(diagonals=[below_diag, main_diag, upper_diag],
               offsets=[-1, 0, 1],
               shape=(N + 1, N + 1),
               format='csr') * (1 / h ** 2)  # Laplacian matrix (with no-flux bc)
    U = np.zeros([N + 1, steps + 1, paths])
    x = np.linspace(-L, L, N + 1) # Spatial mesh
    t = np.linspace(0, T, steps) # Time mesh

    # Set initial condition
    for w in range(paths):
        U[:, 0, w] = U0(x)
    # Euler-Maruyama scheme
    for i in range(steps):
        U[:, i + 1, :] = solve_ivp(U[:, i, :], Ah=Ah, indicator=ind(x))

    # Compute expected value of U(x, t)
    avg_U = np.zeros([N + 1, steps])
    for i in range(N + 1):
        for j in range(steps):
            avg_U[i, j] = np.mean(U[i, j, :])

    # Plot simulation
    boundaries = [-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0]
    cmap = ListedColormap([
        '#ffffff',  # white
        '#ffe6e6',  # very light pink
        '#ffcccc',  # light pink
        '#ff6666',  # medium red
        '#ff3333',  # brighter red
        '#cc0000',  # dark red
        '#800000'  # very dark red
    ])
    norm = BoundaryNorm(boundaries, cmap.N)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    im = ax.imshow(
        avg_U[:,:].T,
        extent=(x[0], x[-1], t[0], t[-1]),
        aspect='auto',
        cmap=cmap,
        norm=norm,
        origin='lower'
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel(r'$x$', fontsize=18)
    ax.set_ylabel(r'$t$', fontsize=18)
    cbar = fig.colorbar(im, ax=ax, ticks=boundaries)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r'$\mathbb{E}[u(x, t)]$', fontsize=18)
    plt.savefig('Plots//hetn_simulation_random.png')

if __name__ == "__main__":
    main()
