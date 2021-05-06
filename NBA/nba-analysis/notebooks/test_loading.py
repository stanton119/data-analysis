# %% Load directly form the data catalogue

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path

metadata = bootstrap_project(Path.cwd().parent)

session = KedroSession.create(project_path=Path.cwd().parent, env=None)
context = session.load_context()

catalog = context.catalog
catalog.list()
df = catalog.load("season_data_2020")
df


# %% Check model distributions
df = catalog.load("shooting_dist")
df
from nba_analysis.pipelines.shooting_per.nodes import BetaBinomial
[BetaBinomial.from_state(df.iloc[idx,1]).plot_pdf() for idx in range(10)]
