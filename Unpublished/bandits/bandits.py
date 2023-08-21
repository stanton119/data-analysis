import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")

np.random.seed(0)


# Arms
class Arm:
    def sample(self, context: list[np.array] = None) -> np.ndarray:
        ...


class DeterministicArm(Arm):
    def __init__(self, reward: float = None) -> None:
        if reward is None:
            reward = np.random.rand(1)[0]
        self.reward = reward

    def __repr__(self) -> str:
        return f"DeterministicArm({self.reward:.3f})"

    def sample(self, context: list[np.array] = None) -> np.ndarray:
        if context is None:
            n_samples = 1
        else:
            n_samples = len(context)
        return np.repeat(self.reward, n_samples)


class BernoulliArm(Arm):
    def __init__(self, p: float = None) -> None:
        if p is None:
            p = np.random.rand(1)[0]
        self.p = p
        self.dist = scipy.stats.bernoulli(p)

    def __repr__(self) -> str:
        return f"Bernoulli({self.p})"

    def sample(self, context: list[np.array] = None) -> np.ndarray:
        if context is None:
            n_samples = 1
        else:
            n_samples = len(context)
        return self.dist.rvs(size=n_samples)


class GaussianArm(Arm):
    def __init__(self, mu: float = None, std:float=None) -> None:
        if mu is None:
            mu = np.random.randn(1)[0]*5
        if std is None:
            std = np.random.rand(1)[0]*10
        self.mu = mu
        self.std = std
        self.dist = scipy.stats.norm(loc=self.mu, scale=self.std)

    def __repr__(self) -> str:
        return f"Gaussian({self.mu},{self.std})"

    def sample(self, context: list[np.array] = None) -> np.ndarray:
        if context is None:
            n_samples = 1
        else:
            n_samples = len(context)
        return self.dist.rvs(size=n_samples)


class BetaArm(Arm):
    def __init__(self, a: float = None, b: float = None) -> None:
        if a is None:
            a = np.random.rand(1)[0]
        if b is None:
            b = np.random.rand(1)[0]
        self.a = a
        self.b = b
        self.dist = scipy.stats.beta(a, b)

    def __repr__(self) -> str:
        return f"Beta({self.a}, {self.b})"

    def sample(self, context: list[np.array] = None) -> np.ndarray:
        if context is None:
            n_samples = 1
        else:
            n_samples = len(context)
        return self.dist.rvs(size=n_samples)

    def plot_pdf(self, ax=None) -> plt.figure:
        x = np.linspace(0, 1, 100)
        y = self.dist.pdf(x)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = None
        ax.plot(x, y)
        return fig


# Policies
class Policy:
    def predict(self, arms: list[list[str]], context: list[list[np.array]]):
        ...

    def update(
        self,
        arms: list[list[str]],
        context: list[list[np.array]],
        rewards: list[list[float]],
    ):
        ...


class RandomPolicy(Policy):
    """
    policy = RandomPolicy(n_arms=2)
    selected_arms, selection_propensities, reward_estimates = policy.predict()
    policy.predict(arms=[[0, 1], [0, 1]])
    policy.update(arms=[[0], [1]], rewards=[[0], [1]])
    """

    def __init__(self, n_arms: int = 4) -> None:
        # a, b are beta priors
        self.n_arms = n_arms

    def predict(
        self, arms: list[list[int]] = None, context: list[list[np.array]] = None
    ):
        if arms is None:
            arms = [range(0, self.n_arms)]
        # ignores context
        # arms = available arms
        selected_arms = []
        selection_propensities = None
        reward_estimates = []

        for _arms in arms:
            _samples = np.random.choice(a=len(_arms), size=len(_arms))
            _selected_arm = _arms[np.argmax(_samples)]

            selected_arms.append([_selected_arm])
            # selection_propensities.append([_samples])  # TODO: needs fixing
            # reward_estimates.append(_samples)

        return selected_arms, selection_propensities, reward_estimates

    def update(
        self,
        arms: list[list[int]] = None,
        context: list[list[np.array]] = None,
        rewards: list[list[float]] = None,
    ):
        pass


class ThompsonSamplingGaussian(Policy):
    """
    policy = ThompsonSamplingGaussian(n_arms=2)
    selected_arms, selection_propensities, reward_estimates = policy.predict()
    policy.predict(arms=[[0, 1], [0, 1]])
    policy.update(arms=[[0], [1]], rewards=[[0], [1]])
    """

    def __init__(self, mu: float = 0.0, std: float = 10.0, n_arms: int = 4) -> None:
        self.mu = np.repeat(mu, n_arms)
        self.std = np.repeat(std, n_arms)
        self.arm_counts = np.repeat(0, n_arms)

    def __repr__(self) -> str:
        return f"Gaussian({self.mu}, {self.std})"

    def predict(
        self, arms: list[list[int]] = None, context: list[list[np.array]] = None
    ):
        if arms is None:
            arms = [range(0, len(self.mu))]
        # ignores context
        # arms = available arms
        selected_arms = []
        selection_propensities = None
        reward_estimates = []

        for _arms in arms:
            _samples = scipy.stats.norm(loc=self.mu[_arms], scale=self.std[_arms]).rvs()
            _selected_arm = _arms[np.argmax(_samples)]

            selected_arms.append([_selected_arm])
            # selection_propensities.append([_samples])  # TODO: needs fixing
            reward_estimates.append(_samples)

        return selected_arms, selection_propensities, reward_estimates

    def update(
        self,
        arms: list[list[int]] = None,
        context: list[list[np.array]] = None,
        rewards: list[list[float]] = None,
    ):
        for _arms, _rewards in zip(arms, rewards):
            self.mu[_arms] = (self.mu[_arms] * self.arm_counts[_arms] + _rewards) / (
                self.arm_counts[_arms] + 1
            )
            self.std[_arms] = np.sqrt(
                np.power(self.std[_arms], 2) * self.arm_counts[_arms] + np.power(np.array(_rewards),2)
            ) / (self.arm_counts[_arms] + 1)
            self.arm_counts[_arms] = self.arm_counts[_arms] + 1

    def plot_pdf(self, ax=None) -> plt.figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = None

        x = np.linspace(-10, 10, 100)  # todo: change to CDF based
        for _idx, (_mu, _std) in enumerate(zip(self.mu, self.std)):
            y = scipy.stats.norm(loc=_mu, scale=_std).pdf(x)
            ax.plot(x, y, label=_idx)
        ax.legend()
        return fig


class ThompsonSamplingBeta(Policy):
    """
    policy = ThompsonSamplingBeta(n_arms=2)
    selected_arms, selection_propensities, reward_estimates = policy.predict()
    policy.predict(arms=[[0, 1], [0, 1]])
    policy.update(arms=[[0], [1]], rewards=[[0], [1]])
    """

    def __init__(self, a: int = 1, b: int = 1, n_arms: int = 4) -> None:
        # a, b are beta priors
        self.a = np.repeat(a, n_arms)
        self.b = np.repeat(b, n_arms)

    def __repr__(self) -> str:
        return f"Beta({self.a}, {self.b})"

    def predict(
        self, arms: list[list[int]] = None, context: list[list[np.array]] = None
    ):
        if arms is None:
            arms = [range(0, len(self.a))]
        # ignores context
        # arms = available arms
        selected_arms = []
        selection_propensities = None
        reward_estimates = []

        for _arms in arms:
            _samples = scipy.stats.beta(self.a[_arms], self.b[_arms]).rvs()
            _selected_arm = _arms[np.argmax(_samples)]

            selected_arms.append([_selected_arm])
            # selection_propensities.append([_samples])  # TODO: needs fixing
            reward_estimates.append(_samples)

        return selected_arms, selection_propensities, reward_estimates

    def update(
        self,
        arms: list[list[int]] = None,
        context: list[list[np.array]] = None,
        rewards: list[list[float]] = None,
    ):
        for _arms, _rewards in zip(arms, rewards):
            self.a[_arms] = self.a[_arms] + _rewards
            self.b[_arms] = self.b[_arms] + 1 - _rewards

    def plot_pdf(self, ax=None) -> plt.figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = None

        x = np.linspace(0, 1, 100)
        for _idx, (_a, _b) in enumerate(zip(self.a, self.b)):
            y = scipy.stats.beta(_a, _b).pdf(x)
            ax.plot(x, y, label=_idx)
        ax.legend()
        return fig


class UCB(Policy):
    """
    policy = UCB(n_arms=2)
    selected_arms, selection_propensities, reward_estimates = policy.predict()
    policy.predict(arms=[[0, 1], [0, 1]])
    policy.update(arms=[[0], [1]], rewards=[[0], [1]])
    """

    def __init__(self, n_arms: int = 4, explore_threshold: float = 2.0) -> None:
        self.arm_counts = np.repeat(0, n_arms)
        self.reward_means = np.repeat(0.0, n_arms)
        self.explore_threshold = explore_threshold

    def __repr__(self) -> str:
        return f"UCB(counts:{self.arm_counts}, reward_means{self.reward_means})"

    def predict(
        self, arms: list[list[int]] = None, context: list[list[np.array]] = None
    ):
        if arms is None:
            arms = [range(0, len(self.arm_counts))]

        # ignores context
        # arms = available arms
        selected_arms = []
        selection_propensities = None
        reward_estimates = []

        for _arms in arms:
            _ucb_values = self.reward_means[_arms] + np.sqrt(
                self.explore_threshold
                * np.log(self.arm_counts[_arms].sum())
                / self.arm_counts[_arms]
            )
            _selected_arm = _arms[np.argmax(_ucb_values)]

            selected_arms.append([_selected_arm])
            # selection_propensities.append([_samples])  # TODO: needs fixing
            reward_estimates.append(self.reward_means[_arms])

        return selected_arms, selection_propensities, reward_estimates

    def update(
        self,
        arms: list[list[int]] = None,
        context: list[list[np.array]] = None,
        rewards: list[list[float]] = None,
    ):
        # update running reward means
        for _arms, _rewards in zip(arms, rewards):
            self.reward_means[_arms] = (
                self.reward_means[_arms] * self.arm_counts[_arms] + _rewards
            ) / (self.arm_counts[_arms] + 1)
            self.arm_counts[_arms] = self.arm_counts[_arms] + 1


# Simulations
