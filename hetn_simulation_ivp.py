import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Type of test
ic = 'tanh'
heterogeneity = 'random'

# Seed for the random heterogeneity
seed = 42

# Physical parameters
mu = 0.1
gamma = 0.01
x0 = -5
delta = 0.5

# Numerical parameters
L = 10.0
N = 500
h = 2 * L / N
T = 500.0
t_steps = 5000
paths = 1


# Set initial condition
def u0(x):
    if ic == 'tanh':
        return np.tanh(x / np.sqrt(2))
    return np.zeros_like(x)


# Set heterogeneity
def S(x):
    indicator = np.logical_and(x >= x0 - delta, x <= x0 + delta).astype(float)
    return np.abs(np.random.normal(loc=0.0, scale=1.0, size=len(x)))*indicator


# Physical function
def f(u, S_vals):
    return u - u ** 3 + gamma + mu * S_vals * u


def rhs(t, u, Ah, S_vals):
    return Ah @ u + f(u, S_vals)


# Plot functions
def plot_heterogeneity(x, S_vals):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, S_vals, 'r')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r'$x$', fontsize=18)
    ax.set_ylabel(r'$s(x,w)$', fontsize=18)
    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.savefig('Plots//' + heterogeneity + '.png')


def plot_simulation(x, t, sol):
    boundaries = [-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0]
    cmap = ListedColormap([
        '#ffffff',  # white
        '#ffe6e6',  # very light pink
        '#ffcccc',  # light pink
        '#ff6666',  # medium red
        '#ff3333',  # brighter red
        '#cc0000',  # dark red
        '#800000'   # very dark red
    ])
    norm = BoundaryNorm(boundaries, cmap.N)
    fig2, ax = plt.subplots(figsize=(9.5, 5))
    im = ax.imshow(
        sol.T,
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
    cbar = fig2.colorbar(im, ax=ax, ticks=boundaries)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r'$\mathbb{E}[U(x, t)]$', fontsize=18)
    simulation = 'Plots//HeN_' + ic + '_' + heterogeneity + '_stable.png'
    plt.savefig(simulation, dpi=300)


def main():
    np.random.seed(seed)
    # Data structures
    x = np.linspace(-L, L, N + 1)
    t = np.linspace(0, T, t_steps)  # Time mesh
    U = np.zeros([N + 1, t_steps, paths])
    main_diag = -2.0 * np.ones(N + 1)
    below_diag = 1.0 * np.ones(N)
    upper_diag = np.copy(below_diag)
    # No-flux Neumann bc
    below_diag[-1] = 2.0
    upper_diag[0] = 2.0
    Ah = diags(diagonals=[below_diag, main_diag, upper_diag],
               offsets=[-1, 0, 1],
               shape=(N + 1, N + 1),
               format='csr') * (1 / h ** 2)
    plot_heterogeneity(x, S(x))

    # Solve PDE using Method of lines and Radau per possible heterogeneity
    for w in range(paths):
        U[:, :, w] = solve_ivp(fun=rhs,
                    t_span=(0, T),
                    y0=u0(x),
                    method='Radau',
                    t_eval=np.linspace(*(0, T), t_steps),
                    args=(Ah, S(x))).y

    # Compute expected value of U(x, t)
    avg_U = np.zeros([N + 1, t_steps])
    for i in range(N + 1):
        for j in range(t_steps):
            avg_U[i, j] = np.mean(U[i, j, :])
    plot_simulation(x, t, avg_U)


if __name__ == '__main__':
    main()