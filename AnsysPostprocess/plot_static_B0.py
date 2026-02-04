# %%
import sys

sys.path.insert(0, "..")
from modules.components import load_data
from modules.graphs import Graph

# %%
file_name = "Line4yamamura_y10_static.csv"

data, time_steps = load_data(file_name)
print(time_steps)
# %%
graph = Graph()

time_step = time_steps[0]
s = data.loc[time_step]
x, y = s.index, s.values
graph.add_plot(x, y)

graph.plot(
    save=True,
    name="static_B0",
    xlabel="Distance along x-axis",
    ylabel="Magnetic Field (T)",
    minorticks=True,
)
