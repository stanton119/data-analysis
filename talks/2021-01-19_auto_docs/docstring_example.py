import pandas as pd


def process_ds(x: pd.DataFrame) -> pd.DataFrame:
    """
    Does something useful to `x`.

    Args:
        x (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: output dataframe
    """
    y = pd.concat([x, x])
    return y


def autodocstring(x: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    y = pd.concat([x, x])
    y["col1"] = y["col1"] * n
    return y


process_ds()

# help(process_ds)
