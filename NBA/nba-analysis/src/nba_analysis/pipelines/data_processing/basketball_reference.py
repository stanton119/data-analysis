import pandas as pd
import numpy as np

from functools import lru_cache


@lru_cache
def get_remote_data(url_type: str, player_id: str = None, season: int = None):
    if url_type == "advanced":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fplayers%2F{player_id[0]}%2F{player_id}.html&div=div_advanced"
    elif url_type == "season_summary_per_game":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fleagues%2FNBA_{season}_per_game.html&div=div_per_game_stats"
    elif url_type == "season_summary_advanced":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fleagues%2FNBA_{season}_advanced.html&div=div_advanced_stats"
    elif url_type == "season_summary_shooting":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fleagues%2FNBA_{season}_shooting.html&div=div_shooting_stats"
    elif url_type == "season_summary_adj_shooting":
        url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fleagues%2FNBA_{season}_adj_shooting.html&div=div_adj-shooting"

    print(f"fetching...{url}")
    df = pd.read_html(
        url,
        flavor="bs4",
    )[0]
    return df


def get_season_summary(season: int = 2020):
    df_per_game = get_remote_data(
        url_type="season_summary_per_game", season=season
    )
    df_advanced = get_remote_data(
        url_type="season_summary_advanced", season=season
    )
    # df_shooting = get_remote_data(url_type="season_summary_adj_shooting", season = season)

    df_per_game = process_df_season_summary(
        df=df_per_game, url_type="season_summary_per_game"
    )
    df_advanced = process_df_season_summary(
        df=df_advanced, url_type="season_summary_advanced"
    )
    # df_shooting = process_df_season_summary(df=df_shooting, url_type="season_summary_adj_shooting")

    merge_cols = [
        col for col in df_per_game.columns if col in df_advanced.columns
    ]
    df_summary = df_per_game.merge(df_advanced, on=merge_cols, how="inner")
    return df_summary


def process_df_totals(df):
    # filter rows/columns and type
    df = df[["Season", "Age", "G", "MP"]]
    df = df.dropna(axis=1, thresh=1)
    df = df.loc[~df.isna().any(axis=1)]
    df["Season"] = pd.to_numeric(df["Season"].apply(lambda x: x[:4]))

    # keep first duplicate for changing team
    df = df.loc[~df["Age"].duplicated()]

    # reindex Age to fill gaps
    age_range = np.arange(df["Age"].min(), df["Age"].max() + 1)
    df = df.set_index("Age").reindex(age_range).reset_index()
    df["Season"] = df["Season"].interpolate()
    df = df.mask(df.isna(), 0)

    # seasonNo as number of seasons played
    df["SeasonNo"] = df["Age"] - df["Age"].min() + 1

    return df


def process_df_season_summary(df, url_type: str):
    """
    Remove player duplicates, take first row for totals
    Some rows not fully populated
    Some columns empty
    """
    # filter columns
    df = df.dropna(axis=1, thresh=1)

    # remove extra header rows
    df = df.loc[df["Rk"] != "Rk"]

    # keep first duplicate for changing team
    df = df.loc[~df[["Rk", "Player"]].duplicated()]

    # numeric types
    str_cols = ["Player", "Pos", "Tm"]
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col])

    if url_type == "season_summary_per_game":
        # replace NaN with medians
        replace_vals = df.median(axis=0, skipna=True)
        for col in replace_vals.index:
            df[col] = df[col].mask(df[col].isna(), replace_vals[col])

    if url_type == "season_summary_advanced":
        # drop minutes played, prefer the MP per game
        df = df.drop(columns="MP")
        # remove NaN rows
        df = df.loc[df.notna().all(axis=1)]

    return df


if 0:

    def _get_remote_data(url):
        print(f"fetching...{url}")
        df = pd.read_html(
            url,
            flavor="bs4",
        )[0]
        return df

    CACHE = dict()

    def get_remote_data(player_id, url_type):
        if url_type == "advanced":
            url = f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fplayers%2F{player_id[0]}%2F{player_id}.html&div=div_advanced"

        if url not in CACHE:
            CACHE[url] = _get_remote_data(url)

        return CACHE[url]


if 0:
    players_dfs = {}
    # %%
    players = {
        "jamesle01": "James",
        "bryanko01": "Bryant",
        "jordami01": "Jordan",
        "abdulka01": "Jabar",
        "chambwi01": "Chamberlain",
        "malonka01": "Malone",
        "duncati01": "Duncan",
        "curryst01": "Curry",
        "paulch01": "Paul",
        "duranke01": "Durrant",
        "hardeja01": "Harden",
        "howardw01": "Howard",
        "westbru01": "Westbrook",
        "rosede01": "Rose",
        "leonaka01": "Leonard",
        "cartevi01": "Carter",
        "antetgi01": "Antetokounmpo",
        "anthoca01": "Anthony",
        "embiijo01": "Embiid",
        "willizi01": "Williamson",
    }

    for player in players:
        if player in players_dfs:
            print(f"{players[player]} already added")
            continue
        print(f"{players[player]} fetching...")
        # df = pd.read_html(f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fplayers%2F{player[0]}%2F{player}.html&div=div_per_game", flavor="bs4")[0]
        df = pd.read_html(
            f"https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fplayers%2F{player[0]}%2F{player}.html&div=div_totals",
            flavor="bs4",
        )[0]

        # filter rows/columns and type
        df = df[["Season", "Age", "G", "MP"]]
        df = df.dropna(axis=1, thresh=1)
        df = df.loc[~df.isna().any(axis=1)]
        df["Season"] = pd.to_numeric(df["Season"].apply(lambda x: x[:4]))

        # keep first duplicate for changing team
        df = df.loc[~df["Age"].duplicated()]

        # reindex Age to fill gaps
        age_range = np.arange(df["Age"].min(), df["Age"].max() + 1)
        df = df.set_index("Age").reindex(age_range).reset_index()
        df["Season"] = df["Season"].interpolate()
        df = df.mask(df.isna(), 0)

        # seasonNo as number of seasons played
        df["SeasonNo"] = df["Age"] - df["Age"].min() + 1
        df["Player"] = players[player]
        players_dfs[player] = df[
            ["Season", "G", "MP", "Age", "SeasonNo", "Player"]
        ]

    df_players = pd.concat(players_dfs, axis=0)

    # Pivot table aligned to career start
    plot_col = "MP"
    df_season_no_plot = df_players[["SeasonNo", "Player", plot_col]].pivot(
        index="Player", columns="SeasonNo", values=plot_col
    )