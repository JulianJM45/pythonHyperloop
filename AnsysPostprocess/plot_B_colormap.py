# %%
import sys

# import numpy as np

sys.path.insert(0, "..")
from modules.components import load_data, load_data2D  # noqa: F401
from modules.graphs import Graph

# %%
file_name = "yam_y10_v0+100.csv"

data, velocities = load_data2D(file_name)
# %%
print(velocities)
# print(time_steps)

data[0].columns


# %%
def slice_airgap2D(v=0, downsample=1):
    df = data[v]
    x, y, mag_B = df["X [m]"].values, df["Y [m]"].values, df["Mag_B [tesla]"].values
    # x_cut = 0.12
    # y_cut = 1
    # mask_x = (x > -x_cut) & (x < x_cut)
    # mask_y = (y > -y_cut) & (y < y_cut)
    # x = x[mask_x & mask_y]
    # y = y[mask_x & mask_y]
    # mag_B = mag_B[mask_x & mask_y]
    if downsample > 1:
        x = x[::downsample]
        y = y[::downsample]
        mag_B = mag_B[::downsample]
    # order = np.argsort(x)
    # x = x[order]
    # mag_B = mag_B[order]
    return x, y, mag_B


# %%
graph = Graph()


# x, mag_B = slice_centerline(t=0)

x, y, mag_B0 = slice_airgap2D(v=0, downsample=1)
x, y, mag_B = slice_airgap2D(v=100, downsample=1)
Delta_B = (mag_B - mag_B0) * 1000
print(len(x), len(y), len(mag_B))
max_B = abs(Delta_B).max()
min_B = -max_B

graph.tripcolor(
    x * 100,
    y * 100,
    Delta_B,
    cbarlabel="ΔB (mT)",
    cmap="seismic",
    vmax=max_B,
    vmin=min_B,
)
# graph.tripcolor(x * 100, y * 100, mag_B0)


graph.plot(
    save=True,
    name="deltaB_2Dprofile_yam_mc_v100",
    # name="B_2Dprofile_yam_mc_v0",
    xlabel="x (cm)",
    ylabel="y (cm)",
    legend=True,
    minorticks=True,
)
