"""
Quick script testing the dask client view to monitor jobs.
"""

import dask
import timeit
from dask.distributed import Client, progress
import time
import numpy as np


def delay_fcn(*args):
    time.sleep(np.random.rand())


if __name__ == "__main__":
    client = Client(n_workers=2, threads_per_worker=2, memory_limit="1GB")
    client
    t1 = timeit.default_timer()
    results_dask = list(map(dask.delayed(delay_fcn), range(100)))
    results_dask = dask.compute(results_dask)[0]
    t2 = timeit.default_timer()
    t_dask = t2 - t1
    print(t_dask)
