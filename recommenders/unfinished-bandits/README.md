# Bandits

## Maths
Setup:

We have a set of possible actions/arms, $\mathcal{A}$, to select from.
Each arm, $a \in \mathcal{A}$, has an associated reward distribution. When we select an arm we receive a reward, $r$, sampled from the reward distribution.
We want to select the arm which maximises the expected reward:
$a^*=\mathrm{argmax}_a E\{r(x,a)\}$. As we only can only observed a subset of rewards, we have to estimate the optimal arm.

Rewards are specified for each arm, $a$, as a function of some features (context vector, $x$): $r(x,a) \in \mathbb{R}^D$.



### Thompson sampling
We estimate the reward distribution for each arm as a context-free distribution. To select an arm, we sample from the estimated reward distributions and select the highest sample. The reward distributions can be updated with observed rewards via Bayes rule.

$r(x,a) = f(a)$.

### Linear Thompson sampling
We estimate the reward distribution for each arm as a linear combination of context features and model weights, $\theta$. It is common to have a single set of model weights and different features for each arm. Features would be created with a common context, $x_c$ and the arm features, $x_a$. We defined $g$ as the function which combines context and arm features: $g(x_c,x_a)$.

$r(x,a) = \langle g(x,a), \theta \rangle$.
The weights distribution can be modelled as a multivariate Gaussian.
We can sample the reward distributions by sampling the weights distribution and constructing a reward. We select the arms with the highest sample.

To update the model weights with Bayes rule, there is an analytical solution in the case of a linear reward, but not in the case of a binary/categorical reward (logistic model).

> [TODO] - how do context vector vary between arms if we assume a single weights vector.

### Arms with context
The reward from a Gaussian context arm can be represented as a linear combination of weights and the context vector.

$r(x,a) = \langle g(x,a), \theta \rangle$.

This can be relaxed to only non-linear function of the context vector.

### Off-policy evaluation
We have a logging policy, $\pi_0$, which was online used to select arms and observe rewards.
We want to evaluate offline a different target policy, $\pi$, on the data saved from the logging policy.

The bandit problem is trying to learn from incomplete data.
Our logging policy only observed the reward from the arms selected. We do not observe the rewards from the non selected arms.
Off policy evaluation is the technique to evaluate a training policy using data which was logged by a different logging policy.

The data we collect from the logging policy is specified as follows. For each round, $i$, we received a context vector, $x_i$, from the available arms, $\mathcal{A}_i$, we selected an arm, $a_i$, which had propensity, $p_i$, and we observed a reward, $r_i$. The set of $n$ rounds is collected as:
$$\mathcal{D} = (a_i, r_i, x_i, \mathcal{A}_i, p_i)_{i=1}^n$$

### Importance weighting
We can use importance weightings to debias the target policy training.

### Replay method
One approach to off-policy evaluation is the [replay method](https://arxiv.org/abs/1003.5956).
Here we select from our target policy, and filter the logged policy data to those rounds where the observed arm matches the selected arm.
We can then update our policy with those actions and repeat for the next batch.
This combines off-policy evaluation and off-policy learning as the policy is updated each batch.
As we don't typically have uniform support for the arms from the logging policy, the regret/performance bounds are typically violated when using replay.
For the same number of rounds, we will observe inflated variance around reward estimations due to the sub-sampled of the observed rewards.

## API
Batch mode is the default API call in each case. If one sample/selection requested, convert to a list first.
Batch mode allows faster computation. Particularly given that we would commonly update the models in batches.

* Arms
  * indexed by str?
    * dict[str,Arm]
  * index by int easier for policies - then no need to know the arm definitions
    * list[Arm]
* Context vectors
  * Data model - list[float] or np.array?
  * np.array for now
  * list of arrays as different number of arms may be available for each round.
  * assume context vector is the same length for all arms
* Policy
  * Tracks all arms in single model
  * How to add/remove arms with time?

### Policy (models)

* predict
  * select which arm to pull in each round within a batch
  * input: arms: list[list[str]], context: list[list[np.array]] or np.array
    * e.g.
      * arms: [[a, b, c], [a, b]]
      * context: [[np.array, np.array, np.array], [np.array, np.array]]
  * output: selected arms: list[list[str]], selection propensities: list[list[float]], reward estimates: list[list[float]]
* update
  * takes selected arms: list[list[str]], associated context vectors, observed rewards and update the model
  * input: arms, context, reward: list[list[float]]

* Thompson sampling
  * takes context
  * linear TS
    * single weights vector, multiple context vectors

### Arms
To simulate/generate data.
Single arm per class or multiple arms per class?
How do we add arms over time?
Dict of arms, dict[str, Arm]

* init
  * with reward function or parameteristed distribution
* sample
  * for a list of context/available arms, returns rewards
  * input: arms: list[int], context: list[float] or np.array?
  * output: rewards: list[float] or np.array?
* expected_reward
  * Arm rewards are stochastic. Having the expected reward allows us to better represent regret.
  * expected_reward(self, context: list[np.array] = None) -> np.ndarray:

### Off-policy evaluation

## TODO
* Off policy evaluation
  * Generate dataset with one policy, train off policy with another
  * inverse propensity scores from Thompson sampling
* Linear arm, linear logistic arm
* Linear Thompson sampling
* General scikit based learn
  * Takes estimator
  * Store data for each arm and refits from scratch at each update call
* Regret
  * Function to get optimal policy, simulate it, find different to cumumlative reward from target policy