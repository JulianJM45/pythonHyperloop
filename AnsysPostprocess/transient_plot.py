# %%
import sys

sys.path.insert(0, "..")
from modules.components import load_data_transient
from modules.graphs import Graph

# %%
file_name = "Force_x_yamamura_long_y10.csv"

time, force = load_data_transient(file_name)

# %%
graph = Graph()
graph.add_plot(time, force)
graph.plot_Graph(
    save=True,
    name="yam_long_force_x",
    # title="Breaking Force over Time",
    xlabel="Time [μs]",
    ylabel="Force_x [mN]",
    minorticks=True,
)
