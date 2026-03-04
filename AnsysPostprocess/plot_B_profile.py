# %%
import sys

import numpy as np

sys.path.insert(0, "..")
from modules.components import load_data, load_data2D  # noqa: F401
from modules.graphs import Graph, get_plot_colors

# %%
file_name = "Line4yamamura_long_y10.csv"
# file_name = "yam_y10_v0+100.csv"

data, time_steps = load_data(file_name)
# data, velocities = load_data2D(file_name)
# %%
# print(velocities)
print(time_steps)

data[0].columns


# %%
def slice_centerline(t=0):
    df = data[t]
    x, mag_B = df["Distance [mm]"].values, df["Mag_B [tesla]"].values
    x_mean = (np.max(x) - np.min(x)) / 2
    x = x - x_mean
    x = x / 1000
    x_cut = 0.1
    mask = (x > -x_cut) & (x < x_cut)
    x = x[mask]
    mag_B = mag_B[mask]
    return x, mag_B


def slice_centerline2D(v=0):
    df = data[v]
    x, y, mag_B = df["X [m]"].values, df["Y [m]"].values, df["Mag_B [tesla]"].values
    x_cut = 0.1
    y_cut = 5e-5
    mask_x = (x > -x_cut) & (x < x_cut)
    mask_y = (y > -y_cut) & (y < y_cut)
    x = x[mask_x & mask_y]
    y = y[mask_x & mask_y]
    mag_B = mag_B[mask_x & mask_y]
    order = np.argsort(x)
    x = x[order]
    mag_B = mag_B[order]
    return x, mag_B


# %%
graph = Graph(width_cm=7.5)

velocities = [0, 100]
colors = get_plot_colors(len(velocities))

# for v, color in zip(velocities, colors):
#     x, mag_B = slice_centerline2D(v=100)
#     graph.add_plot(x * 100, mag_B * 1000, label="B at v=100m/s", color="blue")


x, mag_B0 = slice_centerline(t=0)
# x, mag_B = slice_centerline2D(v=0)
graph.add_plot(x * 100, mag_B0 * 1000, label="B at v=0", color=colors[1])

x, mag_B = slice_centerline(t=9e-4)
# x, mag_B = slice_centerline2D(v=100)
graph.add_plot(x * 100, mag_B * 1000, label="B at v=100m/s", color=colors[0])
# graph.add_scatter(x * 100, mag_B, label="B at v=100m/s", color="blue")
graph.plot(
    save=True,
    name="B_profile_yam_mx_v100_small",
    xlabel="x (cm)",
    ylabel="B (mT)",
    legend=True,
    # minorticks=True,
)

# %%
target_x = 0.0675  # meters
nearest_idx = np.argmin(np.abs(x - target_x))
nearest_x = x[nearest_idx]
nearest_B0 = mag_B0[nearest_idx]
nearest_B = mag_B[nearest_idx]
print(
    f"Nearest point to x={target_x} m is at x={nearest_x:.6f} m with B0={nearest_B0:.6e} T and B={nearest_B:.6e} T, decrease is {(nearest_B0 - nearest_B) / nearest_B0 * 100}% "
)
