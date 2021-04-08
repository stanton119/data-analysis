# %%
"""
Given list of players
Get statistics
Clean
Plot together
"""

# %%
# Data preparation
# %%

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import hvplot.pandas
import holoviews as hv

hv.extension("bokeh")

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
    "willizi01": "Williamson"
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

    # reindex Age
    age_range = np.arange(df["Age"].min(), df["Age"].max() + 1)
    df = df.set_index("Age").reindex(age_range).reset_index()
    df["Season"] = df["Season"].interpolate()
    df = df.mask(df.isna(), 0)

    df["SeasonNo"] = df["Age"] - df["Age"].min() + 1
    df["Player"] = players[player]
    players_dfs[player] = df[["Season", "G", "MP", "Age", "SeasonNo", "Player"]]

df_players = pd.concat(players_dfs, axis=0)

# %%
# Pivot table aligned to career start
plot_col = "MP"
df_season_no_plot = df_players[["SeasonNo", "Player", plot_col]].pivot(
    index="Player", columns="SeasonNo", values=plot_col
)

# %%
# Analysis

# %%
# Compared to other all time greats players Lebron is right up there in terms of minutes played.
# It's only within the last 1/2 seasons that his minutes drop compared to Jabar and Malone.
# Interesting Jordan is an exception due to his two retirement periods.

df_season_no_plot.loc[
    ["James", "Bryant", "Jordan", "Jabar", "Malone"], :
].cumsum(axis=1).transpose().hvplot(grid=True, ylabel="Minutes played")

# %%
# Comparing against superstars form the last couple decades it shows Lebron has managed to avoid any significant injuries affecting his game time.
df_season_no_plot.loc[
    ["James", "Paul", "Howard", "Anthony", "Westbrook", "Rose"], :
].cumsum(axis=1).transpose().hvplot(grid=True, ylabel="Minutes played")

# %%
# Comparing against newer/upcoming superstars:
# For the same number of seasons Kawhi Leondard has played about 50% of the mintues of Lebron James, due to resting during the regular season and injury issues.
# Likewise, Durrant has played about 20% less minutes than Lebron.
# Durrant is the closest to keeping up with Lebron, everyone else is playing less...
# Maybe that means they will keep playing longer?
# I doubt it, mostly this seems down to injury troubles, rather than resting.
# This suggests that no-one in recent years has really been able to stay as healthy as efficient as Lebron James.

df_season_no_plot.loc[
    ["James", "Durrant", "Curry", "Leonard", "Antetokounmpo","Embiid",'Williamson'], :
].cumsum(axis=1).transpose().hvplot(grid=True, ylabel="Minutes played")


# %% Few more players:
df_season_no_plot.cumsum(axis=1).transpose().hvplot(
    grid=True, ylabel="Minutes played"
)

# %%
# Others
# %% Based on age, not season number
plot_col = "MP"
df_age_plot = df_players[["Age", "Player", plot_col]].pivot(
    index="Player", columns="Age", values=plot_col
)
df_age_plot.cumsum(axis=1).transpose().hvplot(
    grid=True, ylabel="Minutes played"
)


# %%
df_season_no_plot.hvplot.heatmap(x="columns", y="index", rot=70, cmap="plasma")
df_age_plot.hvplot.heatmap(x="columns", y="index", rot=70, cmap="plasma")
