# %%
import sys

sys.path.insert(0, "..")
from modules.components import load_data
from modules.graphs import Graph

# %%
file_name = "Line4yamamura_long_y10.csv"

data, time_steps = load_data(file_name)

# %%
graph = Graph()

time_step = time_steps[0]
s = data.loc[time_step]
x, y = s.index, s.values
graph.add_plot(x, y, label="B_0 at t=0")

time_step = time_steps[-1]
s = data.loc[time_step]
x, y = s.index, s.values
graph.add_plot(x, y, label=f"B t={time_step}s", color="blue")

graph.plot(
    save=True,
    name="B_profile_yam_long",
    xlabel="Distance (mm)",
    ylabel="Magnetic Field (T)",
    legend=True,
    minorticks=True,
)
