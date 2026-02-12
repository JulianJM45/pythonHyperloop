# %%
import sys

import numpy as np

sys.path.insert(0, "..")
# from modules.my_plots import compareplot, plot
from modules.graphs import Graph, get_plot_colors

# %%
# velocity
v = 66.6

# constants
mu_0 = 4 * np.pi * 1e-7
# sigma = 10.3e6
sigma = 1e7

L = 2  # full magnet length
d = 0.1  # half rail thickness
a = 0.02  # half rail width
g = 0.015  # air gap

k = mu_0 * sigma * d * v / g
# print(f"K={k}")

B0 = 1


# %%
def main():
    # compare_velocities()
    # longitudinal_distribution()
    # lateral_distribution()
    lift_f_length()
    lift_f_width()
    drag_f_length()
    lift_drag_f_length()

    pass

    # print(B)


# main()

# if __name__ == "__main__":
#     main()


# %%
def longitudinal_distribution():
    graph = Graph(width_cm=7.5, fontsize=10)
    k = mu_0 * sigma * d * v / g
    x_array = np.linspace(-L, L, 10000)

    y_values = np.arange(0, a + 0.01, 0.01)
    colors = get_plot_colors(len(y_values))

    for y, color in zip(y_values, colors):
        b = b_function(x_array, y, k=k)
        graph.add_plot(x_array, b, color=color, label=f"y={y * 100} cm")

    graph.plot_Graph(
        xlabel="x-Position (m)",
        ylabel="Magnetic Field (T)",
        legend=True,
        save=True,
        name="Yam_org_long_dist",
    )


def lateral_distribution():
    graph = Graph(width_cm=7.5, fontsize=10)
    k = mu_0 * sigma * d * v / g
    y_array = np.linspace(-a, a, 100)

    x_values = np.array([0.1 * L, 0.5 * L, 0.9 * L]) - L / 2
    print(f"x values: {x_values}")
    colors = get_plot_colors(len(x_values))

    for x, color in zip(x_values, colors):
        b = b_function(x, y_array, k=k)
        graph.add_plot(y_array * 100, b, color=color, label=f"x={x} m")

    graph.plot_Graph(
        xlabel="y-Position (cm)",
        ylabel="Magnetic Field (T)",
        yrotation=90,
        legend=True,
        save=True,
        name="Yam_org_lat_dist",
    )


def lift_f_length():
    graph = Graph(width_cm=7.5, fontsize=10)
    v_array = np.linspace(0, 100, 101)
    k_array = mu_0 * sigma * d * v_array / g

    l_values = [0.5, 1, 2, 4]
    colors = get_plot_colors(len(l_values))

    for L, color in zip(l_values, colors):
        f_a0 = F_a0(L)
        f_a = F_a(k=k_array, L=L)
        f = f_a / f_a0
        graph.add_plot(v_array, f, color=color, label=f"L={L} m")

    graph.plot_Graph(
        xlabel="Velocity in m/s",
        ylabel=r"$\frac{F_a}{F_{a}^0}$   ",
        yrotation=0,
        # yfontsize=14,
        legend=True,
        save=True,
        name="Yam_org_lift_f_length",
    )


def lift_f_width():
    graph = Graph(width_cm=7.5, fontsize=10)
    v_array = np.linspace(0, 100, 101)
    k_array = mu_0 * sigma * d * v_array / g

    a_values = [0.005, 0.01, 0.015, 0.02]
    colors = get_plot_colors(len(a_values))

    for a, color in zip(a_values, colors):
        f_a0 = F_a0(a=a)
        print(f"F_a0 for a={a}: {f_a0:.4f} N")
        f_a = F_a(k=k_array, a=a)
        print(f"F_a for a={a}: {f_a[1]:.4f} N at v={v_array[1]} m/s")
        f = f_a / f_a0
        graph.add_plot(v_array, f, color=color, label=f"2a={a * 200} cm")

    graph.plot_Graph(
        xlabel="Velocity in m/s",
        ylabel=r"$\frac{F_a}{F_{a}^0}$   ",
        yrotation=0,
        yfontsize=14,
        legend=True,
        save=True,
        name="Yam_org_lift_f_width",
    )


def drag_f_length():
    graph = Graph(width_cm=7.5, fontsize=10)
    v_array = np.linspace(0, 100, 101)
    k_array = mu_0 * sigma * d * v_array / g

    l_values = [0.5, 1, 2, 4]
    colors = get_plot_colors(len(l_values))

    for L, color in zip(l_values, colors):
        f_a0 = F_a0(L)
        f_b = F_b(k=k_array, L=L)
        f = f_b / f_a0
        graph.add_plot(v_array, f, color=color, label=f"L={L} m")

    graph.plot_Graph(
        xlabel="Velocity in m/s",
        ylabel=r"$\frac{F_b}{F_{a}^0}$   ",
        yrotation=0,
        yfontsize=14,
        legend=True,
        save=True,
        name="Yam_org_drag_f_length",
    )


def lift_drag_f_length():
    graph = Graph(width_cm=7.5, fontsize=10)
    v_array = np.linspace(0, 100, 101)
    k_array = mu_0 * sigma * d * v_array / g

    l_values = [0.5, 1, 2, 4]
    colors = get_plot_colors(len(l_values))

    for L, color in zip(l_values, colors):
        f_a = F_a(k=k_array, L=L)
        f_b = F_b(k=k_array, L=L)
        f = f_a / f_b
        graph.add_plot(v_array, f, color=color, label=f"L={L} m")

    graph.plot_Graph(
        xlabel="Velocity in m/s",
        ylabel=r"$\frac{F_a}{F_b}$   ",
        yrotation=0,
        legend=True,
        save=True,
        name="Yam_org_lift-drag_f_length",
    )


def compare_velocities():
    graph = Graph()
    x_array = np.linspace(-L, L, 10000)

    velocities = [0, 30, 66.6, 100]
    colors = get_plot_colors(len(velocities))

    for v, color in zip(velocities, colors):
        k = mu_0 * sigma * d * v / g
        print(f"K={k}")
        b = b_function(x_array, y=0, k=k)
        graph.add_plot(x_array, b, color=color, label=f"v={v} m/s")

    graph.plot_Graph(
        xlabel="x-Position (m)",
        ylabel="Magnetic Field (T)",
        # title="Magnetic Field Profile along x axis",
        legend=True,
        save=True,
        name="Yamamura_org",
    )


def b0_function(x):
    """
    Return 1 where `x` lies in the open interval (-L/2, L/2), else 0.
    Works with scalars and array-like inputs.
    """
    x_arr = np.asarray(x)
    res = np.where((x_arr > -L / 2) & (x_arr < L / 2), 1, 0)

    # If input was a scalar return a Python int, otherwise return the array
    if res.shape == ():
        return int(res)
    return res


def b_function(x, y, k, N=1000):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = np.arange(1, N + 1)
    X_n = Xn(x_arr, n, k)
    lambda__n = lambda_n(n)
    cos_term = np.cos(np.multiply.outer(y_arr, lambda__n))
    summed = np.tensordot(X_n, cos_term, axes=([-1], [-1]))

    base = np.asarray(b0_function(x_arr))
    while base.ndim < summed.ndim:
        base = base[..., np.newaxis]

    result = base + summed
    return np.squeeze(result)


def Xn(x, n, k):
    alpha_n = np.asarray(alpha(n, k))
    beta_n = np.asarray(beta(n, k))
    c_n = np.asarray(c(n))

    if alpha_n.ndim == 2 and alpha_n.shape[1] == 1:
        alpha_n = alpha_n.T
    elif alpha_n.ndim == 1:
        alpha_n = alpha_n[np.newaxis, :]

    if beta_n.ndim == 2 and beta_n.shape[1] == 1:
        beta_n = beta_n.T
    elif beta_n.ndim == 1:
        beta_n = beta_n[np.newaxis, :]

    c_n = c_n[np.newaxis, :]

    x_exp = x[..., np.newaxis]

    left = x_exp < -L / 2
    mid = (x_exp >= -L / 2) & (x_exp < L / 2)
    right = x_exp >= L / 2

    left_val = (
        c_n
        * B0
        * alpha_n
        / (alpha_n - beta_n)
        # attation ! different origin than in the paper: in paper magnet starts at x=0, here x=-L/2
        * (np.exp(-beta_n * (x_exp + L / 2)) - np.exp(-beta_n * (x_exp - L / 2)))
    )
    mid_val = (
        c_n
        * B0
        / (alpha_n - beta_n)
        * (
            beta_n * np.exp(-alpha_n * (x_exp + L / 2))
            - alpha_n * np.exp(-beta_n * (x_exp - L / 2))
        )
    )
    right_val = (
        c_n
        * B0
        * beta_n
        / (alpha_n - beta_n)
        * (np.exp(-alpha_n * (x_exp + L / 2)) - np.exp(-alpha_n * (x_exp - L / 2)))
    )

    res = np.where(left, left_val, 0.0)
    res = np.where(mid, mid_val, res)
    res = np.where(right, right_val, res)
    return res


def alpha(n, k, a=a):
    k_arr = np.asarray(k, dtype=float)
    lambda__n = lambda_n(n, a)[..., np.newaxis]
    return 0.5 * (-k_arr + np.sqrt(k_arr**2 + 4 * lambda__n**2))


def beta(n, k, a=a):
    k_arr = np.asarray(k, dtype=float)
    lambda__n = lambda_n(n, a)[..., np.newaxis]
    return 0.5 * (-k_arr - np.sqrt(k_arr**2 + 4 * lambda__n**2))


def c(n):
    return 4 / (np.pi * (2 * n - 1)) * np.sin((2 * n - 1) * np.pi / 2)


def lambda_n(n, a=a):
    return (2 * n - 1) * np.pi / (2 * a)


def F_a0(L=L, a=a):
    return 2 / mu_0 * a * L * B0**2


def F_a(k=k, L=L, a=a, N=1000):
    n = np.arange(1, N + 1)
    f_a0 = F_a0(L, a)
    c_n = c(n)[:, np.newaxis]
    f_n = f(n, k, L, a)
    return f_a0 * np.sum(c_n**2 * f_n, axis=0)


def f(n, k, L, a=a):
    alpha_n = alpha(n, k, a)
    beta_n = beta(n, k, a)
    alpha_dash = alpha_n / (alpha_n - beta_n)
    beta_dash = beta_n / (alpha_n - beta_n)

    addend1 = L + beta_dash / alpha_n * (2 + beta_dash) * (1 - np.exp(-alpha_n * L))
    addend2 = alpha_dash / beta_n * (2 - alpha_dash) * (1 - np.exp(beta_n * L))
    addend3 = (
        2
        * alpha_dash
        * beta_dash
        / (alpha_n + beta_n)
        * (np.exp(-alpha_n * L) - np.exp(beta_n * L))
    )
    return 1 / (2 * L) * (addend1 + addend2 + addend3)


def F_b(k, L, a=a, N=1000):
    n = np.arange(1, N + 1)
    f_a0 = F_a0(L)
    c_n = c(n)[:, np.newaxis]
    alpha_n = alpha(n, k, a)
    beta_n = beta(n, k, a)
    alpha_dash = alpha_n / (alpha_n - beta_n)
    beta_dash = beta_n / (alpha_n - beta_n)

    return (
        f_a0
        * g
        / L
        * np.sum(
            c_n**2
            * (
                alpha_dash * (np.exp(beta_n * L) - 1)
                + beta_dash * (np.exp(-alpha_n * L) - 1)
            ),
            axis=0,
        )
    )


if __name__ == "__main__":
    main()
