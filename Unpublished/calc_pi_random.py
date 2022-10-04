# %% calculate pi from random numbers
import numpy as np


def estimate_pi(n: int = 1000000) -> float:
    x = np.random.rand(n, 2)
    z = np.sqrt(np.sum(np.power(x, 2), axis=1))
    return np.sum(z < 1) / len(z) * 4

estimate_pi(1000000)
