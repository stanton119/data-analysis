# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use("seaborn-whitegrid")


class BetaBinomial:
    def __init__(self, alpha=1, beta=1) -> None:
        self.alpha = alpha
        self.beta = beta

    def bayes_update(self, trials, successes):
        self.alpha = self.alpha + successes
        self.beta = self.beta + trials - successes

    def pdf(self) -> Tuple[np.array, np.array]:
        from scipy.stats import beta

        x = np.linspace(0.0, 1.0, 100)
        z = beta.pdf(x, a=self.alpha, b=self.beta)

        return x, z

    def plot_pdf(self, **kwargs):
        import matplotlib.pyplot as plt

        x, z = self.pdf()
        ax = plt.plot(x, z, **kwargs)
        return ax

    def output_state(self):
        return [self.alpha, self.beta]

    @classmethod
    def from_state(cls, state):
        return cls(state[0], state[1])


def shooting_per_model(
    df: pd.DataFrame, prior_beta: List[float] = None
) -> pd.DataFrame:
    df["FT_Scored"] = (df["FT"] * df["G"]).round()
    df["FT_Attempted"] = (df["FTA"] * df["G"]).round()

    # for each player find posterior beta for shooting %
    # we take the prob of scoring a FT as a bernouli trial
    # that bernouli trial prob comes from a beta distribution
    # As we are looking over a whole season, we have multi bernouli trials
    # This is represented by a sample from a binomial distribution
    dists = df.groupby(["Player"]).apply(fit_beta_per_player)

    return dists.reset_index()


def fit_beta_per_player(
    df: pd.DataFrame, prior_beta: List[float] = None
) -> BetaBinomial:
    df_temp = df.sort_values("Age")
    if prior_beta is not None:
        dist = BetaBinomial(alpha=prior_beta[0], beta=prior_beta[1])
    else:
        dist = BetaBinomial()
    for idx, row in df_temp.iterrows():
        dist.bayes_update(
            trials=row["FT_Attempted"], successes=row["FT_Scored"]
        )
    return dist.output_state()


def plot_shooting_per(df: pd.DataFrame, no_players: int = 10):
    df
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in np.random.choice(a=df.shape[0], size=no_players, replace=False):
        x, y = BetaBinomial.from_state(df.iloc[idx, 1]).pdf()
        ax.plot(x, y, label=df["Player"].iloc[idx])
    ax.set_xlabel("Shooting %")
    ax.set_ylabel("PDF()")
    fig.legend()
    return fig
