"""
Make http requests in parallel using multiprocessing and dask

Which API is nicer

Timing each suggests that dask/multiprocessing, as expected, speed up the requests by running in parallel.

It might be preferrable to run in an asychronos manner instead of parallel processing.
"""

import multiprocessing
import timeit
from datetime import datetime
import time

import dask
import pandas as pd
import requests


def fetch_results(date):
    date_str = date.strftime("%Y-%m-%d")
    r = requests.get(f"https://api.carbonintensity.org.uk/intensity/date/{date_str}")
    return r.json()


def fetch_results_mock(date):
    time.sleep(0.2)
    return date


if __name__ == "__main__":
    fcn = fetch_results
    fcn = fetch_results_mock
    params = pd.date_range(datetime(2020, 1, 1), periods=20).tolist()

    # for loop
    t1 = timeit.default_timer()
    results_for = list()
    for param in params:
        results_for.append(fcn(param))
    t2 = timeit.default_timer()
    t_for = t2 - t1
    print(f"For loop: {t_for:.1f}")

    # map
    t1 = timeit.default_timer()
    results_map = list(map(fcn, params))
    t2 = timeit.default_timer()
    t_map = t2 - t1
    print(f"Map: {t_map:.1f}")

    # multiprocessing
    # Doesnt run in ipython - https://stackoverflow.com/questions/48846085/python-multiprocessing-within-jupyter-notebook
    mp = multiprocessing.Pool(8)
    t1 = timeit.default_timer()
    results_mp = mp.map(fcn, params)
    t2 = timeit.default_timer()
    t_mp = t2 - t1
    print(f"Multiprocessing: {t_mp:.1f}")

    # dask
    t1 = timeit.default_timer()
    results_dask = list(map(dask.delayed(fcn), params))
    results_dask = dask.compute(results_dask)[0]
    t2 = timeit.default_timer()
    t_dask = t2 - t1
    print(f"Dask: {t_dask:.1f}")

    # check results
    assert results_for == results_map
    assert results_for == results_mp
    assert results_for == results_dask
