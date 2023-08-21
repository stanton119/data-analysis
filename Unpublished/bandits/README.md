# Bandits

## Maths
Setup:

We have $D$ possible actions/arms to select from.
Each arm, $a$, has an associated reward distribution, $r$.
Rewards are specified for each arm, $a$, as a function of some features (context vector, $x$): $r(x,a) \in D$.

The optimal arm is the one that maximises the expected reward:
$a^*=\mathrm{argmax}_a E\{r(x,a)\}$.

As we only can only observed a subset of rewards, we have to estimate the optimal arm.

### Thompson sampling
We estimate the reward distribution for each arm as a context-free distribution. To select an arm, we sample from the reward distributions and select the highest sample. The reward distributions can updated with observed rewards via Bayes rule.

$r(x,a) = f(a)$.

### Linear Thompson sampling
We estimate the reward distribution for each arm as a linear combination of context features and model weights, $\theta$. It is common to have a single set of model weights and different features for each arm. Features would be created with a common context and the arm, $g(x,a)$.

$r(x,a) = \langle g(x,a), \theta \rangle$.
The weights distribution can be modelled as a multivariate Gaussian.
We can sample the reward distributions by sampling the weights distribution and constructing a reward. We select the arms with the highest sample.

To update the model weights with Bayes rule, there is an analytical solution in the case of a linear reward, but not in the case of a binary/categorical reward (logistic model).

### Off-policy evaluation


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
  * takes selected arms: list[list[str]], observed rewards and update the model
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

### Off-policy evaluation

## TODO
* Thompson sampling
* Gaussian policy updates
* Linear arm, linear logistic arm
* Linear Thompson sampling
* Off policy evaluation
  * inverse propensity scores from Thompson sampling
