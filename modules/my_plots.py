import matplotlib.pyplot as plt


def plot(X, Y):
    plt.plot(X, Y)
    plt.xlabel("x")
    plt.ylabel("B")
    plt.title("Magnetic Field")
    plt.grid(True)
    plt.show()


def compareplot(X, Y1, Y2, label1="model 1", label2="model 2"):
    plt.plot(X, Y1, label=label1, color="blue", linestyle="-")
    # plt.scatter(X, Y2, label=label2, color="red", marker="x")
    plt.plot(X, Y2, label=label2, color="red", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("B")
    plt.title("Magnetic Field Comparison")
    plt.grid(True)
    plt.legend()
    plt.show()
