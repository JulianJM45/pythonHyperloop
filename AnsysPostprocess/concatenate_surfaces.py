# %%
import os
import sys

import pandas as pd

sys.path.insert(0, "..")

import matplotlib.pyplot as plt
import numpy as np

# from modules.graphs import Graph

# %%

workdir = "../../Simulation_results/Mechanical"
file = "file.csv"
filepath = os.path.join(workdir, file)
# %%
df = pd.read_csv(filepath, sep="\t")
df.columns
# %%

x = df["X Coordinate (m)"].values
y = df["Y Coordinate (m)"].values
mag_B = df["Total Magnetic Flux Density ()"].values


# %%
mask_y = (y > -0.005) & (y < 0.005)
x = x[mask_y]
y = y[mask_y]
mag_B = mag_B[mask_y]
# %%
mask_x = (x > -0.1) & (x < 0.1)
x = x[mask_x]
y = y[mask_x]
mag_B = mag_B[mask_x]
# %%
vmax = 0.97 * mag_B.max()
vmin = 0.91 * mag_B.max()
# plt.tricontour(x, y, mag_B, levels=200, vmin=vmin, vmax=vmax)
# plt.tripcolor(x, y, mag_B, vmin=vmin, vmax=vmax)
plt.tripcolor(x, y, mag_B, vmin=vmin, vmax=vmax, shading="flat")

plt.colorbar()
plt.show()

# %%
y_cut = 1e-3
mask_y0 = (y > -y_cut) & (y < y_cut)
y0 = y[mask_y0]
x0 = x[mask_y0]
mag_B0 = mag_B[mask_y0]
order = np.argsort(x0)
x0 = x0[order]
y0 = y0[order]
mag_B0 = mag_B0[order]
print(y0)
# %%

plt.plot(x0, mag_B0)
plt.show()
