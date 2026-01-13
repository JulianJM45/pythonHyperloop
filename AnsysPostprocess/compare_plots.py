# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce

# %%
working_directory = "../Simulation_results/"
# csv_file_names = ["line2_100ms_2e-5_1e-6.csv", "line2_100ms_2e-5_1e-5.csv"]
csv_file_names = ["Line4yamamura_y10.aedt.csv"]
file_name = csv_file_names[0]
data = {}
time_steps = []
for index, file_name in enumerate(csv_file_names):
    df = pd.read_csv(working_directory + file_name, sep=";")
    # df = pd.read_csv(working_directory + file_name, sep=',')
    data[index] = df.pivot(
        index="Time [s]", columns="Distance [mm]", values="Mag_B (Real) [tesla]"
    )
    time_steps.append(df["Time [s]"].unique())
time_steps = reduce(np.intersect1d, time_steps)
# %%
# time_step = 10e-05
time_step = time_steps[10]
plt.figure(figsize=(8, 4))
for index in data.keys():
    s = data[index].loc[time_step]
    x = s.index
    y = s.values
    plt.plot(x, y, label=f"Simulation {index + 1}")

# plt.plot(x, y, marker='', linestyle='-')
plt.xlabel("Distance [mm]")
plt.ylabel("Mag_B [tesla]")
plt.title(f"Magnetic field at Time = {time_step} s")
plt.grid()
plt.show()
# %%
time_step = 0
for time_step in time_steps:
    y1 = data[0].loc[time_step].values
    y2 = data[1].loc[time_step].values
    msd = np.nanmean((y1 - y2) ** 2)
    print(f"Mean square difference in time_step {time_step} :", msd)
# %%
start0 = 4
end0 = 25
velocity = 100e3
data_set = data[0]
# time_step = time_steps[10]

distances = []
mag_Bs = []
time_steps = data_set.index
for index, time_step in enumerate(time_steps):
    # start_point = start0 + velocity * time_step
    # end_point = end0 + velocity * time_step
    # distance = data_set.columns[(start_point <= data_set.columns) & (data_set.columns <= end_point)]
    # distances.append(distance)
    # mag_Bs.append(data_set.loc[time_step, distance])

    mag_Bs.append(data_set.loc[time_step, :])
    if index > 0:
        msd = np.nanmean((mag_Bs[index] - mag_Bs[index - 1]) ** 2)
        print(
            f"Mean square difference between time_step {time_steps[index - 1]} and {time_step} :",
            msd,
        )

    # if index > 30: break


# plt.figure(figsize=(8, 4))
# plt.plot(distance, mag_B, label=f'Simulation 1')
