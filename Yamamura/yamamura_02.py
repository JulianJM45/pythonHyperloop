# %%
import sys

import numpy as np
from scipy.fft import fft, fftfreq, ifft

sys.path.insert(0, "..")
from modules.graphs import Graph, get_plot_colors
from modules.interpolation import getSpline

# from modules.my_plots import compareplot

# %%
# velocity
v = 100

# constants
mu_0 = 4 * np.pi * 1e-7
sigma = 10.3e6

L = 10  # magnet length
N = 1000  # Number of points in the domain
# l = 5  # magnet lengt
d = 0.035  # rail width
a = 0.1  # rail thickness
g = 0.1  # air gap


n = 1000  # order
dx = 4 * L / N  # Spatial step size
x = np.arange(-2 * L, 2 * L, dx)
omega = 2 * np.pi * fftfreq(N, d=dx)

# B0 = 1
# B0 = getSpline(x)
# X_B0, Y_B0 = load_data()
cs = getSpline()

k = mu_0 * sigma * d * v / g
# k = mu_0 * sigma * d * v
k = -200
print(f"K={k}")


# %%


def main():
    graph = Graph()
    velocities = [0, 50, 100, 200]
    colors = get_plot_colors(len(velocities))

    for v, color in zip(velocities, colors):
        # k = mu_0 * sigma * d * v / g
        k = mu_0 * sigma * d * v / (g + d)
        print(f"K={k}")
        b = yam_02(k=k)
        graph.add_plot(x, b, color=color, label=f"v={v} m/s")

    graph.plot_Graph(
        xlabel="Position (m)",
        ylabel="Magnetic Field (T)",
        title="Magnetic Field Profile along x axis",
        legend=True,
        save=True,
        name="Yamamura_02",
    )


main()


# %%
def yam_02(k=100):
    # b0 = b01_function(x)
    b0 = b02_function(x)
    b = b_function(b0, k)
    return b


def b01_function(x):
    """
    Return 1 where `x` lies in the open interval (-l/2, l/2), else 0.
    Works with scalars and array-like inputs.
    """
    x_arr = np.asarray(x)
    res = np.where((x_arr > -L) & (x_arr < L), 1, 0)

    # If input was a scalar return a Python int, otherwise return the array
    if res.shape == ():
        return int(res)
    return res


def b02_function(x):
    """Return a spline function derived from simulation data"""
    return np.nan_to_num(cs(x), nan=0.0)


def b_function(b0, k):
    b0hat = fft(b0)
    bi = np.zeros_like(x)
    bi = np.sum([Xn(b0hat, i, k) for i in range(n)], axis=0)

    return bi + b0


def Xn(b0hat, n, k):
    # print('xn executed')
    c_n = c(n)
    lambda_n = lambda_function(n)
    numerator = c_n * (omega**2 + 1j * omega * k)
    denominator = omega**2 + 1j * omega * k + lambda_n**2
    return ifft(-numerator / denominator * b0hat)


def c(n):
    return 4 / (np.pi * (2 * n - 1)) * np.sin((2 * n - 1) * np.pi / 2)


def lambda_function(n):
    return (2 * n - 1) * np.pi / (2 * a)


# %%

if __name__ == "__main__":
    main()
