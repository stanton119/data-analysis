"""
Make http requests in parallel using multiprocessing and dask

Which API is nicer
Speed differences?
"""

import multiprocessing
import timeit
from datetime import datetime

import dask
import pandas as pd
import requests


def fetch_results(date):
    date_str = date.strftime("%Y-%m-%d")
    r = requests.get(
        f"https://api.carbonintensity.org.uk/intensity/date/{date_str}"
    )
    return r.json()


if __name__ == "__main__":
    date_range = pd.date_range(datetime(2020, 1, 1), periods=20).tolist()

    # for loop
    t1 = timeit.default_timer()
    results_for = list()
    for date in date_range:
        results_for.append(fetch_results(date))
    t2 = timeit.default_timer()
    t_for = t2 - t1
    print("For loop: ", t_for)

    # map
    t1 = timeit.default_timer()
    results_map = list(map(fetch_results, date_range))
    t2 = timeit.default_timer()
    t_map = t2 - t1
    print("Map: ", t_map)

    # multiprocessing
    # Doesnt run in ipython - https://stackoverflow.com/questions/48846085/python-multiprocessing-within-jupyter-notebook
    mp = multiprocessing.Pool(4)
    t1 = timeit.default_timer()
    results_mp = mp.map(fetch_results, date_range)
    t2 = timeit.default_timer()
    t_mp = t2 - t1
    print("Multiprocessing: ", t_mp)

    # dask
    t1 = timeit.default_timer()
    results_dask = list(map(dask.delayed(fetch_results), date_range))
    results_dask = dask.compute(results_dask)[0]
    t2 = timeit.default_timer()
    t_dask = t2 - t1
    print("Dask: ", t_dask)

    # check results
    assert results_for == results_map
    assert results_for == results_mp
    assert results_for == results_dask
