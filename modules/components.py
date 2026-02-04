from functools import reduce

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .graphs import Graph

working_directory = "../../Simulation_results/"


def load_data(file_names, working_directory=working_directory):
    # Track if input was a single string
    single_file = isinstance(file_names, str)

    if single_file:
        file_names = [file_names]

    data = {}
    time_steps = []
    for index, file_name in enumerate(file_names):
        df = pd.read_csv(working_directory + file_name, sep=";")
        # df = pd.read_csv(working_directory + file_name, sep=',')
        data[index] = df.pivot(
            index="Time [s]", columns="Distance [mm]", values="Mag_B [tesla]"
        )
        time_steps.append(df["Time [s]"].unique())
    time_steps = reduce(np.intersect1d, time_steps)

    if single_file:
        return data[0], time_steps

    return data, time_steps


def load_data_transient(file_name, working_directory=working_directory):
    df = pd.read_csv(working_directory + file_name, sep=",")
    time, force = df["Time [us]"].values, df["Force_x [mNewton]"].values
    return time, force


def merge_csv_files(file_name1, file_name2, working_directory=working_directory):
    """
    Merge two CSV files together.

    Args:
        file_name1: First CSV file name
        file_name2: Second CSV file name
        working_directory: Directory where CSV files are located

    Returns:
        merged_df: Merged DataFrame with combined data from both files
        time_steps: Common time steps present in both files
    """
    # Load both CSV files
    df1 = pd.read_csv(working_directory + file_name1, sep=";")
    df2 = pd.read_csv(working_directory + file_name2, sep=";")

    # Pivot each dataframe separately
    pivoted_df1 = df1.pivot(
        index="Time [s]", columns="Distance [mm]", values="Mag_B [tesla]"
    )
    pivoted_df2 = df2.pivot(
        index="Time [s]", columns="Distance [mm]", values="Mag_B [tesla]"
    )

    # Merge the pivoted dataframes by combining rows (union of time steps) and columns (union of distances)
    # Use outer join to include all time steps and distances from both files
    merged_df = pd.concat([pivoted_df1, pivoted_df2], axis=0)

    # Remove duplicate rows if any (same time step)
    merged_df = merged_df[~merged_df.index.duplicated(keep="first")]

    # Sort by time
    merged_df = merged_df.sort_index()

    # Get unique time steps
    time_steps = merged_df.index.values

    return merged_df, time_steps


def plot_data(data, time_step):
    graph = Graph()

    # Handle both DataFrame (single file) and dict (multiple files)
    if isinstance(data, pd.DataFrame):
        # Single file case: data is a DataFrame
        s = data.loc[time_step]
        x = s.index
        y = s.values
        graph.add_plot(x, y, label="B Profile")
    else:
        # Multiple files case: data is a dict of DataFrames
        for index in data.keys():
            s = data[index].loc[time_step]
            x = s.index
            y = s.values
            graph.add_plot(x, y, label="B Profile for index " + str(index))

    graph.plot_Graph(
        legend=True,
        minorticks=True,
        title="Magnetic Field Profile at Time Step " + str(time_step),
        xlabel="Distance [mm]",
        ylabel="Magnetic Field [T]",
    )


def get_xy(s):
    return s.index, s.values


def get_b0(data, time_step, x_interval):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    s = data.loc[time_step]
    x, y = get_xy(s)
    mask = (x >= x_interval[0]) & (x <= x_interval[1])
    B0 = y[mask].mean()

    return B0


def s_function(x, x0=0, y0=0, amp=1, a=1):
    return amp / (1 + np.exp(-a * (x - x0))) + y0


def test_B1(data, time_step, x0=40, x1=26, amp=0.036, a=0.2):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    s = data.loc[time_step]
    x, y = get_xy(s)
    y0 = np.interp(x1, x, y)
    B1 = s_function(x, x0=x0, y0=y0, amp=amp, a=a)
    graph = Graph()
    graph.add_plot(x, y, label="B Profile at time_step " + str(time_step), color="blue")
    graph.add_plot(x, B1, label="test B Profile ", color="orange")
    graph.plot_Graph(
        legend=False,
        minorticks=True,
        title="Test Magnetic Field Profile at Time Step " + str(time_step),
        xlabel="Distance [mm]",
        ylabel="Magnetic Field [T]",
    )


def fit_B1(data, time_step, B0, fitrange=[26, 68], x0=40, amp=0.036, a=0.2):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    s = data.loc[time_step]
    x, y = get_xy(s)
    fit_mask = (x >= fitrange[0]) & (x <= fitrange[1])
    x_fit, y_fit = x[fit_mask], y[fit_mask]
    y0 = np.interp(fitrange[0], x, y)

    popt, pcov = curve_fit(s_function, x_fit, y_fit, p0=[x0, y0, amp, a])
    x0_fit, y0_fit, amp_fit, a_fit = popt

    eindringtiefe_x = 2 * (x0_fit - 25)
    dip_y = B0 - y0_fit
    b_increase = y0_fit + amp_fit - B0

    print(f"Fitted parameters:")
    print(f"x0 = {x0_fit:.4f}")
    print(f"y0 = {y0_fit:.4f}")
    print(f"amp = {amp_fit:.4f}")
    print(f"a = {a_fit:.4f}")
    print(f"Fit quality: {np.sqrt(np.diag(pcov))}")

    print(f"Eindringtiefe_x:{eindringtiefe_x:.4f}mm")
    print(f"Dip_y:{dip_y:.4f}T, that is {(dip_y / B0 * 100):.4f}%")
    print(f"B increase:{b_increase:.4f}T, that is {(b_increase / B0 * 100):.4f}%")

    B_fitted = s_function(x, x0_fit, y0_fit, amp_fit, a_fit)

    graph = Graph()
    graph.add_plot(x, y, label="B Profile at time_step " + str(time_step), color="blue")
    graph.add_plot(x, B_fitted, label="fitted B Profile ", color="orange")
    graph.plot_Graph(
        legend=False,
        minorticks=True,
        title="Test Magnetic Field Profile at Time Step " + str(time_step),
        xlabel="Distance [mm]",
        ylabel="Magnetic Field [T]",
    )

    return eindringtiefe_x, dip_y, b_increase
