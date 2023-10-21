from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import urllib.request
import zipfile
from tqdm import trange

DATA_PATH = Path(__file__).parents[1] / "data"


# generation functions
def gen_dummy_data():
    if 1:
        df = pd.DataFrame()
        df["ds"] = pd.date_range(
            start="2010-01-01", end="2025-01-01", freq="1D"
        )
        df["y"] = np.random.rand(df.shape[0], 1)
        df["x1"] = np.random.rand(df.shape[0], 1)
    else:
        data_location = "https://raw.githubusercontent.com/ourownstory/neural_prophet/master/"
        df = pd.read_csv(
            data_location + "example_data/wp_log_peyton_manning.csv"
        )

    df_train = df.iloc[: int(df.shape[0] / 2)]
    df_test = df.iloc[int(df.shape[0] / 2) :]


def get_weather_data() -> pd.DataFrame:
    try:
        with open(DATA_PATH / "weather_data.csv.zip", "r") as f:
            pass
    except FileNotFoundError:
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        res = urllib.request.urlretrieve(
            url, filename=DATA_PATH / "weather_data.csv.zip"
        )

    df = pd.read_csv(DATA_PATH / "weather_data.csv.zip")
    df = df.rename(columns={"Date Time": "ds", "T (degC)": "y"})
    df["ds"] = pd.to_datetime(df["ds"], format=r"%d.%m.%Y %H:%M:%S")

    filt = (df["ds"].dt.hour==0) & (df["ds"].dt.minute==0)
    df = df.loc[filt]
    df = df.drop_duplicates()
    r = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq="D")
    df = df.set_index("ds").reindex(r).rename_axis("ds").reset_index()
    df['y'] = df['y'].fillna(method='ffill')

    return df.set_index("ds")[["y"]]


def gen_ar_data(freqs: List[float] = [10, 3]) -> pd.DataFrame:
    df = pd.DataFrame()
    df["ds"] = pd.date_range(start="2010-01-01", end="2025-01-01", freq="1D")

    for freq in freqs:
        df[f"x{freq}"] = np.sin(
            np.linspace(start=0, stop=freq * 2 * np.math.pi, num=df.shape[0])
        )
    df["y"] = df.iloc[:, 1:].sum(axis=1)

    return df.set_index("ds")


def gen_multivar_ar_data(freqs: List[float] = [10, 3]) -> pd.DataFrame:
    df = pd.DataFrame()
    df["ds"] = pd.date_range(start="2010-01-01", end="2025-01-01", freq="1D")

    weights = np.random.rand(3)

    for freq in freqs:
        df[f"x{freq}"] = np.sin(
            np.linspace(start=0, stop=freq * 2 * np.math.pi, num=df.shape[0])
        )
    df["y"] = df.iloc[:, 1:].sum(axis=1)

    return df.set_index("ds")


def get_stock_data() -> pd.DataFrame:
    df = pd.read_csv("data/aapl_us.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df = df.drop(columns=["OpenInt"])
    return df.set_index("ds")


def get_energy_data() -> pd.DataFrame:
    try:
        with open(DATA_PATH / "energy_data" / "LD2011_2014.txt", "r") as f:
            pass
    except FileNotFoundError:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
        res = urllib.request.urlretrieve(url, filename="data/energy_data.zip")
        with zipfile.ZipFile(DATA_PATH / "energy_data.zip", "r") as zip_ref:
            zip_ref.extractall(DATA_PATH / "energy_data")

    df = pd.read_csv(
        DATA_PATH / "energy_data" / "LD2011_2014.txt",
        sep=";",
        index_col=0,
        parse_dates=True,
        decimal=",",
    )

    return df
