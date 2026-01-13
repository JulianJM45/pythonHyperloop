import numpy as np
import matplotlib.pyplot as plt

# velocity
v = 100

# constants
mu_0 = 4 * np.pi * 1e-7
sigma = 10.3e6

L = 20  # magnet length
d = 0.1  # rail width
a = 0.1  # rail thickness
g = 0.02  # air gap

B0 = 1

K = mu_0 * sigma * d * v / g
# K = 100
print(f"K={K}")


def main():
    x_array = np.linspace(-L, 2 * L, 1000)
    # x_array = [1]
    B = []
    for x in x_array:
        B.append(b(x, z=0))
    # B = np.array(B)
    # print(B)
    plot(x_array, B)


def B_0(x):
    if 0 <= x <= L:
        return B0
    else:
        return 0


def b(x, z, N=8):
    n = np.arange(1, N + 1)  # Shape (N, 1)
    X_n = Xn(x, n)
    lambda__n = lambda_n(n)
    return B_0(x) + B0 * np.sum(X_n * np.cos(lambda__n * z))


def Xn(x, n):
    # print(f"x={x}")
    alpha_n = alpha(n)
    # print(f"alpha({n})={alpha_n}")
    beta_n = beta(n)
    # print(f"beta({n})={beta_n}")
    # c_n = c(n)
    # print(f"c({n})={c_n}")

    if x < 0:
        return (
            alpha_n
            / (alpha_n - beta_n)
            * (np.exp(beta_n * x) - np.exp(beta_n * (x - L)))
        )
    elif x < L:
        return (
            1
            / (alpha_n - beta_n)
            * (beta_n * np.exp(alpha_n * x) - alpha_n * np.exp(beta_n * (x - L)))
        )
    else:
        return (
            beta_n
            / (alpha_n - beta_n)
            * (np.exp(alpha_n * x) - np.exp(alpha_n * (x - L)))
        )


def alpha(n):
    return 0.5 * (K - np.sqrt(K**2 + 4 * lambda_n(n) ** 2))


def beta(n):
    return 0.5 * (K + np.sqrt(K**2 + 4 * lambda_n(n) ** 2))


def c(n):
    return 4 / (np.pi * (2 * n - 1)) * np.sin((2 * n - 1) * np.pi / 2)


def a_n(n):
    return 3 * np.sin(lambda_n(n)) / (np.pi * (n - 0.5))


def lambda_n(n):
    return (n - 0.5) * np.pi / (0.5 * a)


def plot(X, Y):
    plt.plot(X, Y)
    plt.xlabel("x")
    plt.ylabel("B")
    plt.title("Magnetic Field")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
