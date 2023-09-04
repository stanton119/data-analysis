from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import tqdm

plt.style.use("seaborn-whitegrid")

np.random.seed(0)

# Type definitions
ArmsList = list[int]
AvailableArms = list[ArmsList]  # length can change per round
ContextVectors = list[list[np.array]]  # length can change per round
SelectedArms = list[ArmsList]  # fix number per round
SelectionPropensities = list[np.array]  # length can change per round
RewardEstimates = list[np.array]  # length can change per round
Rewards = list[list[float]]  # fix number per round


# Arms
class Arm:
    def sample(self, context: list[np.array] = None) -> np.ndarray:
        ...

    def expected_reward(self, context: list[np.array] = None) -> np.ndarray:
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

    def expected_reward(self, context: list[np.array] = None) -> np.ndarray:
        return self.sample(context=context)


class BernoulliArm(Arm):
    def __init__(self, p: float = None) -> None:
        if p is None:
            p = np.random.rand(1)[0]
        self.p = p
        self.dist = scipy.stats.bernoulli(p)

    def __repr__(self) -> str:
        return f"Bernoulli({self.p:.3f})"

    def sample(self, context: list[np.array] = None) -> np.ndarray:
        if context is None:
            n_samples = 1
        else:
            n_samples = len(context)
        return self.dist.rvs(size=n_samples)

    def expected_reward(self, context: list[np.array] = None) -> np.ndarray:
        if context is None:
            n_samples = 1
        else:
            n_samples = len(context)
        return np.repeat(self.p, n_samples)


class GaussianArm(Arm):
    """
    The reward distribution is given by a Gaussian distribution.
    This is equivalent to the contextual version as:
    ```python
    arm = bandits.GaussianContextArm(context_len=0)
    ```
    """

    def __init__(self, mu: float = None, std: float = None) -> None:
        if mu is None:
            mu = np.random.randn(1)[0] * 5
        if std is None:
            std = np.random.rand(1)[0] * 10
        self.mu = mu
        self.std = std
        self.dist = scipy.stats.norm(loc=self.mu, scale=self.std)

    def __repr__(self) -> str:
        return f"Gaussian({self.mu:.3f},{self.std:.3f})"

    def sample(self, context: list[np.array] = None) -> np.ndarray:
        if context is None:
            n_samples = 1
        else:
            n_samples = len(context)
        return self.dist.rvs(size=n_samples)

    def expected_reward(self, context: list[np.array] = None) -> np.ndarray:
        if context is None:
            n_samples = 1
        else:
            n_samples = len(context)
        return np.repeat(self.mu, n_samples)

    def plot_pdf(self, ax=None, label: str = None) -> plt.figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = None

        x_lim = self.dist.ppf(q=[0.001, 0.999])
        x = np.linspace(x_lim[0], x_lim[1], 100)
        y = self.dist.pdf(x)
        y = y / y.max()
        ax.plot(x, y, label=label)
        return fig


class GaussianContextArm(Arm):
    def __init__(
        self,
        bias: float = None,
        weights: np.array = None,
        context_len: int = 5,
        std: float = None,
    ) -> None:
        if weights is None:
            weights = np.random.randn(context_len)
        if bias is None:
            bias = np.random.randn(1)[0] * 5
        if std is None:
            std = np.random.rand(1)[0] * 10

        self.bias = bias
        self.weights = weights
        self.std = std

    def __repr__(self) -> str:
        return f"Gaussian({self.bias:.3f},{self.weights},{self.std:.3f})"

    def sample(self, context: list[np.array] = None) -> np.ndarray:
        if context is None:
            context = np.random.randn(1, len(self.weights))

        context = np.array(context)
        return (
            self.bias
            + np.dot(context, self.weights)
            + self.std * np.random.randn(context.shape[0])
        )

    def expected_reward(self, context: list[np.array] = None) -> np.ndarray:
        if context is None:
            context = np.random.randn(len(self.weights))

        return self.bias + np.dot(context, self.weights)


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

    def expected_reward(self, context: list[np.array] = None) -> np.ndarray:
        raise NotImplementedError

    def plot_pdf(self, ax=None) -> plt.figure:
        x = np.linspace(0, 1, 100)
        y = self.dist.pdf(x)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = None
        ax.plot(x, y)
        return fig


# Policies
class Policy:
    def __init__(self, n_arms: int = 4, top_k: int = 1) -> None:
        self.n_arms = n_arms
        self.top_k = top_k

    def predict(
        self, arms: AvailableArms = None, context: ContextVectors = None
    ) -> tuple[SelectedArms, SelectionPropensities, RewardEstimates]:
        """
        arms: 1 list per round. In each round we have a list of available arms.
        context: 1 list per round. In each round we have a list of context vectors, 1 per available arm.
        """
        ...

    def update(
        self,
        arms: SelectedArms = None,
        context: ContextVectors = None,
        rewards: Rewards = None,
    ):
        """
        we expect only selected arms, not all available arms.
        context and rewards dimensions match the selected arms.
        """
        ...

    def logging_callback(self) -> dict:
        """Called after each training cycle to log policy state."""

        return {}


class RandomPolicy(Policy):
    """
    Random prob of reward for each arm. Given some reward_probs
    ```python
    policy = RandomPolicy(n_arms=2)
    selected_arms, selection_propensities, reward_estimates = policy.predict()
    policy.predict(arms=[[0, 1], [0, 1]])
    policy.update(arms=[[0], [1]], rewards=[[0], [1]])
    ```
    """

    def __init__(
        self,
        n_arms: int = 4,
        reward_probs: np.array = None,
        uniform: bool = True,
        top_k: int = 1,
    ) -> None:
        self.n_arms = n_arms
        if reward_probs is None:
            if uniform:
                reward_probs = np.full(shape=self.n_arms, fill_value=1 / self.n_arms)
            else:
                reward_probs = np.random.uniform(size=self.n_arms)

        if reward_probs.sum() != 1.0:
            print("scaling reward_probs")
            reward_probs = reward_probs / reward_probs.sum()

        self.reward_probs = reward_probs

        super().__init__(n_arms=n_arms, top_k=top_k)

    def predict(
        self, arms: AvailableArms = None, context: ContextVectors = None
    ) -> tuple[SelectedArms, SelectionPropensities, RewardEstimates]:
        if arms is None:
            arms = [range(0, self.n_arms)]
        # ignores context
        # arms = available arms
        selected_arms = []
        selection_propensities = []
        reward_estimates = []

        for _arms in arms:
            _selected_arm_arg = np.random.choice(
                a=_arms, size=self.top_k, replace=False, p=self.reward_probs[_arms]
            )
            _selected_arm = [_arms[_sel] for _sel in _selected_arm_arg]
            selected_arms.append(_selected_arm)

            selection_propensities.append(self.reward_probs[_arms])
            # reward_estimates.append(_samples)

        return selected_arms, selection_propensities, reward_estimates

    def update(
        self,
        arms: SelectedArms = None,
        context: ContextVectors = None,
        rewards: Rewards = None,
    ):
        pass


class ThompsonSamplingGaussian(Policy):
    """
    policy = ThompsonSamplingGaussian(n_arms=2)
    selected_arms, selection_propensities, reward_estimates = policy.predict()
    policy.predict(arms=[[0, 1], [0, 1]])
    policy.update(arms=[[0], [1]], rewards=[[0], [1]])
    """

    def __init__(
        self, mu: float = 0.0, std: float = 10.0, n_arms: int = 4, top_k: int = 1
    ) -> None:
        self.mu = np.repeat(mu, n_arms)
        self.std = np.repeat(std, n_arms)
        self.arm_counts = np.repeat(0, n_arms)
        super().__init__(n_arms=n_arms, top_k=top_k)

    def __repr__(self) -> str:
        return f"Gaussian({self.mu}, {self.std})"

    def _get_dists(self):
        return [
            scipy.stats.norm(loc=_mu, scale=_std)
            for _mu, _std in zip(self.mu, self.std)
        ]

    def predict(
        self, arms: AvailableArms = None, context: ContextVectors = None
    ) -> tuple[SelectedArms, SelectionPropensities, RewardEstimates]:
        if arms is None:
            arms = [range(0, self.n_arms)]
        # ignores context
        # arms = available arms
        selected_arms = []
        selection_propensities = None
        reward_estimates = []

        _dists = self._get_dists()

        for _arms in arms:
            _samples = np.array([_dist.rvs() for _dist in _dists])

            # find top_k samples
            _arm_order = np.flip(np.argsort(_samples))
            _selected_arm_arg = _arm_order[: self.top_k]
            _selected_arm = [_arms[_sel] for _sel in _selected_arm_arg]
            selected_arms.append(_selected_arm)

            # selection_propensities.append([_samples])  # TODO: needs fixing
            reward_estimates.append(_samples)

        return selected_arms, selection_propensities, reward_estimates

    def update(
        self,
        arms: SelectedArms = None,
        context: ContextVectors = None,
        rewards: Rewards = None,
    ):
        for _arms, _rewards in zip(arms, rewards):
            prev_var = np.power(self.std[_arms], 2)
            prev_mean = self.mu[_arms]
            prev_count = self.arm_counts[_arms]
            new_val = np.array(_rewards)

            new_mean = (prev_mean * prev_count + new_val) / (prev_count + 1)
            # bug?
            new_var = (
                (prev_var + np.power(prev_mean, 2)) * prev_count + np.power(new_val, 2)
            ) / (prev_count + 1) - np.power(new_mean, 2)
            new_std = np.sqrt(new_var)

            self.mu[_arms] = new_mean
            self.std[_arms] = new_std
            self.arm_counts[_arms] = self.arm_counts[_arms] + 1

    def plot_pdf(self, ax=None) -> plt.figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = None

        _dists = self._get_dists()

        for _idx, _dist in enumerate(_dists):
            x_lim = _dist.ppf(q=[0.001, 0.999])
            x = np.linspace(x_lim[0], x_lim[1], 100)
            y = _dist.pdf(x)
            y = y / y.max()
            ax.plot(x, y, label=_idx)
        ax.legend()
        return fig

    def logging_callback(self) -> dict:
        return {
            "mu": self.mu,
            "std": self.std,
            "arm_counts": self.arm_counts,
            "quantiles": self.get_quantiles(),
        }

    def get_quantiles(self, quantiles=None) -> np.array:
        """Returns quantiles as [arms x quantiles]."""

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        dists = self._get_dists()
        return np.vstack([dist.ppf([quantiles]) for dist in dists])


class ThompsonSamplingBeta(Policy):
    """
    policy = ThompsonSamplingBeta(n_arms=2)
    selected_arms, selection_propensities, reward_estimates = policy.predict()
    policy.predict(arms=[[0, 1], [0, 1]])
    policy.update(arms=[[0], [1]], rewards=[[0], [1]])
    """

    def __init__(self, a: int = 1, b: int = 1, n_arms: int = 4, top_k: int = 1) -> None:
        # a, b are beta priors
        self.a = np.repeat(a, n_arms)
        self.b = np.repeat(b, n_arms)
        super().__init__(n_arms=n_arms, top_k=top_k)

    def __repr__(self) -> str:
        return f"Beta({self.a}, {self.b})"

    def _get_dists(self):
        return [scipy.stats.beta(a, b) for a, b in zip(self.a, self.b)]

    def predict(
        self, arms: AvailableArms = None, context: ContextVectors = None
    ) -> tuple[SelectedArms, SelectionPropensities, RewardEstimates]:
        if arms is None:
            arms = [range(0, len(self.a))]
        # ignores context
        # arms = available arms
        selected_arms = []
        selection_propensities = None
        reward_estimates = []

        for _arms in arms:
            # construct as a multivariate dist is 10x faster
            _samples = scipy.stats.beta(self.a[_arms], self.b[_arms]).rvs()

            # find top_k samples
            _arm_order = np.flip(np.argsort(_samples))
            _selected_arm_arg = _arm_order[: self.top_k]
            _selected_arm = [_arms[_sel] for _sel in _selected_arm_arg]
            selected_arms.append(_selected_arm)

            # selection_propensities.append([_samples])  # TODO: needs fixing
            reward_estimates.append(_samples)

        return selected_arms, selection_propensities, reward_estimates

    def update(
        self,
        arms: SelectedArms = None,
        context: ContextVectors = None,
        rewards: Rewards = None,
    ):
        for _arms, _rewards in zip(arms, rewards):
            self.a[_arms] = self.a[_arms] + _rewards
            self.b[_arms] = self.b[_arms] + 1 - _rewards

    def plot_pdf(self, ax=None) -> plt.figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = None

        dists = self._get_dists()
        x = np.linspace(0, 1, 100)
        for _idx, _dist in enumerate(dists):
            y = _dist.pdf(x)
            ax.plot(x, y, label=_idx)
        ax.legend()
        return fig

    def logging_callback(self) -> dict:
        return {"a": self.a, "b": self.b, "quantiles": self.get_quantiles()}

    def get_quantiles(self, quantiles=None) -> np.array:
        """Returns quantiles as [arms x quantiles]."""

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        dists = self._get_dists()
        return np.vstack([dist.ppf([quantiles]) for dist in dists])


class UCB(Policy):
    """
    policy = UCB(n_arms=2)
    selected_arms, selection_propensities, reward_estimates = policy.predict()
    policy.predict(arms=[[0, 1], [0, 1]])
    policy.update(arms=[[0], [1]], rewards=[[0], [1]])
    """

    def __init__(
        self, n_arms: int = 4, explore_threshold: float = 2.0, top_k: int = 1
    ) -> None:
        self.arm_counts = np.repeat(0, n_arms)
        self.reward_means = np.repeat(0.0, n_arms)
        self.explore_threshold = explore_threshold
        super().__init__(n_arms=n_arms, top_k=top_k)

    def __repr__(self) -> str:
        return f"UCB(counts:{self.arm_counts}, reward_means{self.reward_means})"

    def predict(
        self, arms: AvailableArms = None, context: ContextVectors = None
    ) -> tuple[SelectedArms, SelectionPropensities, RewardEstimates]:
        if arms is None:
            arms = [range(0, len(self.arm_counts))]

        # ignores context
        # arms = available arms
        selected_arms = []
        selection_propensities = None
        reward_estimates = []

        for _arms in arms:
            _arm_counts = np.maximum(
                self.arm_counts[_arms], np.full(fill_value=1, shape=len(_arms))
            )
            _ucb_values = self.reward_means[_arms] + np.sqrt(
                self.explore_threshold * np.log(_arm_counts.sum()) / _arm_counts
            )

            # find top_k samples
            _arm_order = np.flip(np.argsort(_ucb_values))
            _selected_arm_arg = _arm_order[: self.top_k]
            _selected_arm = [_arms[_sel] for _sel in _selected_arm_arg]
            selected_arms.append(_selected_arm)

            # selection_propensities.append([_samples])  # TODO: needs fixing
            reward_estimates.append(self.reward_means[_arms])

        return selected_arms, selection_propensities, reward_estimates

    def update(
        self,
        arms: SelectedArms = None,
        context: ContextVectors = None,
        rewards: Rewards = None,
    ):
        # update running reward means
        for _arms, _rewards in zip(arms, rewards):
            self.reward_means[_arms] = (
                self.reward_means[_arms] * self.arm_counts[_arms] + _rewards
            ) / (self.arm_counts[_arms] + 1)
            self.arm_counts[_arms] = self.arm_counts[_arms] + 1

    def logging_callback(self) -> dict:
        return {
            "arm_counts": self.arm_counts,
            "reward_means": self.reward_means,
        }


class LinearThompsonSamplingGaussian(Policy):
    """
    policy = LinearThompsonSamplingGaussian(n_arms=2)
    selected_arms, selection_propensities, reward_estimates = policy.predict(context=...)
    policy.predict(arms=[[0, 1], [0, 1]])
    policy.update(arms=[[0], [1]], rewards=[[0], [1]])
    """

    def __init__(
        self, mu: float = 0.0, std: float = 10.0, n_arms: int = 4, top_k: int = 1
    ) -> None:
        self.mu = np.repeat(mu, n_arms)
        self.std = np.repeat(std, n_arms)
        self.arm_counts = np.repeat(0, n_arms)
        super().__init__(n_arms=n_arms, top_k=top_k)

    def __repr__(self) -> str:
        return f"Gaussian({self.mu}, {self.std})"

    def _get_dists(self):
        return [
            scipy.stats.norm(loc=_mu, scale=_std)
            for _mu, _std in zip(self.mu, self.std)
        ]

    def predict(
        self, arms: AvailableArms = None, context: ContextVectors = None
    ) -> tuple[SelectedArms, SelectionPropensities, RewardEstimates]:
        if arms is None:
            arms = [range(0, self.n_arms)]
        # ignores context
        # arms = available arms
        selected_arms = []
        selection_propensities = None
        reward_estimates = []

        _dists = self._get_dists()

        for _arms in arms:
            _samples = np.array([_dist.rvs() for _dist in _dists])

            # find top_k samples
            _arm_order = np.flip(np.argsort(_samples))
            _selected_arm_arg = _arm_order[: self.top_k]
            _selected_arm = [_arms[_sel] for _sel in _selected_arm_arg]
            selected_arms.append(_selected_arm)

            # selection_propensities.append([_samples])  # TODO: needs fixing
            reward_estimates.append(_samples)

        return selected_arms, selection_propensities, reward_estimates

    def update(
        self,
        arms: SelectedArms = None,
        context: ContextVectors = None,
        rewards: Rewards = None,
    ):
        for _arms, _rewards in zip(arms, rewards):
            prev_var = np.power(self.std[_arms], 2)
            prev_mean = self.mu[_arms]
            prev_count = self.arm_counts[_arms]
            new_val = np.array(_rewards)

            new_mean = (prev_mean * prev_count + new_val) / (prev_count + 1)
            new_var = (
                (prev_var + np.power(prev_mean, 2)) * prev_count + np.power(new_val, 2)
            ) / (prev_count + 1) - np.power(new_mean, 2)
            new_std = np.sqrt(new_var)

            self.mu[_arms] = new_mean
            self.std[_arms] = new_std
            self.arm_counts[_arms] = self.arm_counts[_arms] + 1

    def plot_pdf(self, ax=None) -> plt.figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = None

        _dists = self._get_dists()

        for _idx, _dist in enumerate(_dists):
            x_lim = _dist.ppf(q=[0.001, 0.999])
            x = np.linspace(x_lim[0], x_lim[1], 100)
            y = _dist.pdf(x)
            y = y / y.max()
            ax.plot(x, y, label=_idx)
        ax.legend()
        return fig


# Simulations
@dataclass
class SimulationResults:
    available_arms: AvailableArms
    selected_arms: SelectedArms
    selection_propensities: SelectionPropensities
    reward_estimates: RewardEstimates
    rewards: Rewards


def simulate_batch(
    policy: Policy,
    arms: list[Arm],
    available_arms: AvailableArms = None,
    context: ContextVectors = None,
    batch_size: int = 10,
) -> SimulationResults:
    """
    Assumes fixed arm availability for each round if no available_arms given.
    batch_size - number of rounds to simulate.

    TODO fix context to each arm sample method
    """

    if available_arms is None:
        n_arms = len(arms)
        available_arms = [range(0, n_arms)] * batch_size

    # select from policy, observe rewards and update policy
    selected_arms, selection_propensities, reward_estimates = policy.predict(
        arms=available_arms, context=context
    )
    rewards = [
        [arms[_idx].sample()[0] for _idx in _selected_arms]
        for _selected_arms in selected_arms
    ]
    if context is not None:
        selected_context = []
        for _round in batch_size:
            selected_context.append(
                [context[_round][_idx] for _idx in selected_arms[_round]]
            )
    else:
        selected_context = None
    policy.update(arms=selected_arms, context=selected_context, rewards=rewards)

    return SimulationResults(
        available_arms=available_arms,
        selected_arms=selected_arms,
        selection_propensities=selection_propensities,
        reward_estimates=reward_estimates,
        rewards=rewards,
    )


def simulate_batches(
    policy: Policy,
    arms: list[Arm],
    available_arms: AvailableArms = None,
    context: ContextVectors = None,
    n_batches: int = 10,
    batch_size: int = 10,
    batch_callback: Callable[[Policy, int], dict] = None,
) -> SimulationResults:
    _available_arms = []
    selected_arms = []
    selection_propensities = []
    reward_estimates = []
    rewards = []
    regret = []

    for _batch_id in tqdm.trange(n_batches):
        _batch_idx = (_batch_id * batch_size, (_batch_id + 1) * batch_size)

        batch_results = simulate_batch(
            policy=policy,
            arms=arms,
            available_arms=available_arms[_batch_idx[0] : _batch_idx[1]]
            if available_arms is not None
            else None,
            context=context[_batch_idx[0] : _batch_idx[1]]
            if context is not None
            else None,
            batch_size=batch_size,
        )
        # append results
        if batch_results.available_arms is None:
            _available_arms = None
        else:
            _available_arms = _available_arms + batch_results.available_arms
        selected_arms = selected_arms + batch_results.selected_arms
        if batch_results.selection_propensities is None:
            selection_propensities = None
        else:
            selection_propensities = (
                selection_propensities + batch_results.selection_propensities
            )
        if batch_results.reward_estimates is None:
            reward_estimates = None
        else:
            reward_estimates = reward_estimates + batch_results.reward_estimates
        rewards = rewards + batch_results.rewards

        if batch_callback is not None:
            batch_callback(policy, _batch_id)

    return SimulationResults(
        available_arms=_available_arms,
        selected_arms=selected_arms,
        selection_propensities=selection_propensities,
        reward_estimates=reward_estimates,
        rewards=rewards,
    )

def get_optimal_policy():
    """Get optimal policy arms. Find reward/expected reward.
    Find regret as difference to simulation."""

    raise NotImplementedError
    # regret if case of known arms
    # find best arm
    expected_rewards = [
        [arms[_arm].expected_reward(context=context)[0] for _arm in _available_arms]
        for _available_arms in available_arms
    ]

    # find top_k samples
    _arm_order = np.flip(np.argsort(expected_rewards))
    _selected_arm_arg = _arm_order[: policy.top_k]
    _selected_arm = [_arms[_sel] for _sel in _selected_arm_arg]
    selected_arms.append(_selected_arm)

    # get max expected reward
    # diff with observed reward
    rewards

    # get max expected reward
    expected_rewards = [
        [arms[_arm].expected_reward()[0] for _arm in _available_arms]
        for _available_arms in available_arms
    ]
    max_expected_reward = np.max(expected_rewards, axis=1)
    # diff with observed reward
    max_expected_reward
    rewards

def log_policy_quantiles(policy: Policy, n: int, policy_params: dict):
    """Logs policy params for each batch:

    ```python
    policy_params = {}
    simulation_results = bandits.simulate_batches(
        policy=policy, arms=arms, n_batches=20, batch_size=10, return_regret=False, batch_callback=lambda policy, n: log_policy_quantiles(policy, n, policy_params)
    )
    ```
    """
    policy_params[n] = policy.logging_callback()


# results plotting
def plot_arm_selection(selected_arms: SelectedArms, ax=None) -> plt.figure:
    # if in multiple batches, combine together to single long list
    if not isinstance(selected_arms[0][0], int):
        raise NotImplementedError

    # plot arm selection
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = None

    # cumulative sum selected arms by arm index
    selected_arms = np.array(selected_arms)
    selected_arms_idx = np.zeros(
        shape=(selected_arms.shape[0], selected_arms.max() + 1)
    )
    for _k in range(selected_arms.shape[1]):
        selected_arms_idx[np.arange(selected_arms.shape[0]), selected_arms[:, _k]] = 1
    selected_arms_cumsum = selected_arms_idx.cumsum(axis=0)

    # add preceeding row to start counts from 0
    selected_arms_cumsum = np.vstack(
        [np.zeros(shape=(1, selected_arms.max() + 1)), selected_arms_cumsum]
    )

    x = np.arange(selected_arms_cumsum.shape[0])
    for _idx in range(selected_arms_cumsum.shape[1]):
        ax.plot(x, selected_arms_cumsum[:, _idx], label=_idx)
    ax.set(
        title="Arm selection against round", xlabel="Round number", ylabel="Arm count"
    )
    ax.legend()
    return fig


def plot_rewards(rewards: Rewards, ax=None, label: str = None) -> plt.figure:
    # plot arm selection
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = None

    # cumulative sum selected arms by arm index
    rewards = np.array(rewards)
    rewards = rewards.sum(axis=1)
    rewards_cumsum = rewards.cumsum(axis=0)
    # add preceeding row to start counts from 0
    rewards_cumsum = np.append(0, rewards.cumsum())

    x = np.arange(rewards_cumsum.shape[0])
    ax.plot(x, rewards_cumsum, label=label)
    ax.set(
        title="Observed rewards against round",
        xlabel="Round number",
        ylabel="Cumulative reward",
    )
    return fig


def plot_bandit_quantiles(
    bandit_quantiles: dict[int, np.array], arms_expected_reward=None, ax=None
):
    bandit_quantiles_idx = list(bandit_quantiles.keys())
    bandit_quantiles = list(bandit_quantiles.values())
    n_arms, n_quantiles = bandit_quantiles[0].shape

    # convert to tall dataframe
    bandit_quantiles_np = np.vstack(bandit_quantiles)
    bandit_quantiles_df = pd.DataFrame(
        bandit_quantiles_np, columns=[f"q{idx+1}" for idx in range(n_quantiles)]
    )
    bandit_quantiles_df["trial_set"] = np.repeat(bandit_quantiles_idx, n_arms) + 1
    bandit_quantiles_df["bandit"] = np.tile(
        range(n_arms), len(bandit_quantiles)
    ).astype(str)
    bandit_quantiles_df = (
        bandit_quantiles_df.set_index(["trial_set", "bandit"])
        .stack()
        .rename("reward")
        .reset_index()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None
    sns.lineplot(
        x="trial_set", y="reward", hue="bandit", data=bandit_quantiles_df, ax=ax
    )

    if arms_expected_reward is not None:
        ax.hlines(
            y=arms_expected_reward,
            xmin=bandit_quantiles_df["trial_set"].min(),
            xmax=bandit_quantiles_df["trial_set"].max(),
            linestyles="--",
            alpha=0.5,
        )
    return ax, bandit_quantiles_df


# utils
def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Calculate softmax function.

    ```python
    softmax(np.array([0.1, 0.3, 0.2]))
    softmax(np.array([[0.5, 0.1, 0.2], [0.1, 0.3, 0.2]]))
    ```
    """
    x = x * temp
    flatten = False
    if x.ndim == 1:
        flatten = True
        x = x[np.newaxis, :]

    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    if flatten:
        return (numerator / denominator).flatten()
    else:
        return numerator / denominator
