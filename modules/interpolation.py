# %%
import pandas as pd
from scipy.interpolate import CubicSpline


def load_data():
    data = pd.read_csv("coil_magnet_line2.csv", delimiter=";")
    # data = pd.read_csv("../coil_magnet_line2.csv", delimiter=";")
    X = data["Distance [mm]"].values
    Y = data["Mag_B [tesla]"].values
    # print("Smallest Y value:", Y.min())
    offset = Y.min()
    Y = Y - offset
    Y[0] = Y[-1] = 0
    # Shift X so that data is symmetric to the y-axis
    center = (X.max() + X.min()) / 2
    X = X - center
    return X, Y


def getSpline():
    X, Y = load_data()

    cs = CubicSpline(X, Y, bc_type="natural", extrapolate=False)
    # Y_smooth = cs(x)
    # Y_smooth = np.nan_to_num(Y_smooth, nan=0.0)

    # return Y_smooth
    return cs


if __name__ == "__main__":
    X, Y = load_data()
    # Y_smooth = getSpline(X)
    cs = getSpline()
    Y_smooth = cs(X)
    # Y_smooth = np.nan_to_num(Y_smooth, nan=0.0)

    # print(getSpline(502))

    from my_plots import *

    plot(X, Y_smooth)
    # compareplot(X, Y, Y_smooth, label1="data", label2="model")
