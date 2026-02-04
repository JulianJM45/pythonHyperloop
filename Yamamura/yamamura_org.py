# %%
import sys

import numpy as np

sys.path.insert(0, "..")
# from modules.my_plots import compareplot, plot
from modules.graphs import Graph, get_plot_colors

# %%
# velocity
v = 100

# constants
mu_0 = 4 * np.pi * 1e-7
sigma = 10.3e6

L = 1  # magnet length
d = 0.035  # rail width
a = 0.1  # rail thickness
g = 0.1  # air gap

B0 = 1

# k = mu_0 * sigma * d * v / g
# k = mu_0 * sigma * d * v
# k = -200
# print(f"K={k}")


# %%
def main():
    graph = Graph()
    x_array = np.linspace(-2 * L, 2 * L, 10000)

    velocities = [0, 50, 100, 200]
    colors = get_plot_colors(len(velocities))

    for v, color in zip(velocities, colors):
        k = mu_0 * sigma * d * v / g
        print(f"K={k}")
        b = yam_org(k=k)
        graph.add_plot(x_array, b, color=color, label=f"v={v} m/s")

    graph.plot_Graph(
        xlabel="Position (m)",
        ylabel="Magnetic Field (T)",
        title="Magnetic Field Profile along x axis",
        legend=True,
        save=True,
        name="Yamamura_org",
    )

    # print(B)


main()

# if __name__ == "__main__":
#     main()


# %%
def yam_org(k=100):
    x_array = np.linspace(-2 * L, 2 * L, 10000)
    # b_0 = b0_function(x_array)
    # plot(x_array, b_0)
    # x_array = [1]
    b_list = []
    for i in range(len(x_array)):
        x_val = float(x_array[i])
        b_list.append(b_function(x_val, z=0, k=k))
    b_array = np.array(b_list, dtype=np.float64)
    return b_array


def B_0(x):
    if -L <= x < L:
        return B0
    else:
        return 0


def b0_function(x):
    """
    Return 1 where `x` lies in the open interval (-l, l), else 0.
    Works with scalars and array-like inputs.
    """
    x_arr = np.asarray(x)
    res = np.where((x_arr > -L) & (x_arr < L), 1, 0)

    # If input was a scalar return a Python int, otherwise return the array
    if res.shape == ():
        return int(res)
    return res


def b_function(x, z, k, N=800):
    n = np.arange(1, N + 1)  # Shape (N, 1)
    X_n = Xn(x, n, k)
    lambda__n = lambda_n(n)
    return B_0(x) + float(np.sum(X_n * np.cos(lambda__n * z)))


def Xn(x, n, k):
    alpha_n = alpha(n, k)
    beta_n = beta(n, k)
    c_n = c(n)

    if x < -L:  # left of magnet
        return (
            c_n
            * B0
            * alpha_n
            / (alpha_n - beta_n)
            * (np.exp(-beta_n * (x + L)) - np.exp(-beta_n * (x - L)))
        )
    elif x < L:  # inside magnet
        return (
            c_n
            * B0
            / (alpha_n - beta_n)
            * (
                beta_n * np.exp(-alpha_n * (x + L))
                - alpha_n * np.exp(-beta_n * (x - L))
            )
        )
    else:  # right of magnet (x >= L)
        return (
            c_n
            * B0
            * beta_n
            / (alpha_n - beta_n)
            * (np.exp(-alpha_n * (x + L)) - np.exp(-alpha_n * (x - L)))
        )


def alpha(n, k):
    return 0.5 * (-k + np.sqrt(k**2 + 4 * lambda_n(n) ** 2))


def beta(n, k):
    return 0.5 * (-k - np.sqrt(k**2 + 4 * lambda_n(n) ** 2))


def c(n):
    return 4 / (np.pi * (2 * n - 1)) * np.sin((2 * n - 1) * np.pi / 2)


def lambda_n(n):
    return (2 * n - 1) * np.pi / (2 * a)


if __name__ == "__main__":
    main()
