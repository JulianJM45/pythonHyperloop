# %%
import numpy as np

n = np.arange(1, 1000001)


def c(n):
    return 4 / (np.pi * (2 * n - 1)) * np.sin((2 * n - 1) * np.pi / 2)


print(np.sum(c(n) ** 2))
