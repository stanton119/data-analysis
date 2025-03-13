# Maths

## Random projection dimensionality reduction
This is useful when the feature space is large.
PCA is computationally expensive as its based on the eigendecomposition (inverse operation).
PCA can be sensitive to outliers as its a linear transform (non robust).
Random projections are fast as we dont calculate the projection matrix based on the data.
We generate it from random gaussians. The number of components for the projection matrix is typically taken from the [Johnson-Lindenstrauss lemma](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_johnson_lindenstrauss_bound.html#sphx-glr-auto-examples-miscellaneous-plot-johnson-lindenstrauss-bound-py).
This states that the pairwise distance between points in the component space is similar to the original space if the number of components is chosen relating to the log of the number of samples.

This is extended through the sparse random projection, which suggests similar results, but with only integer random variables [-1, 0, 1], instead of a gaussian. This is faster for computation again.

[More recent approaches](https://arxiv.org/pdf/2005.00511.pdf) suggest using random projections to first reduce dimensionality and then applying PCA to the result.

## Causal inference

References
* Causal textbook - [Causal Inference for The Brave and True](https://matheusfacure.github.io/python-causality-handbook/)
* Practical walkthrough - [Be Careful When Interpreting Predictive Models in Search of Causal Insights
](https://towardsdatascience.com/be-careful-when-interpreting-predictive-models-in-search-of-causal-insights-e68626e664b6)
* Economics textbook - [QuantEcon](https://datascience.quantecon.org/applications/heterogeneity.html)


### Double ML
In double ML we typically:
1. residualise $Y$ with $Z$ to give $\hat{Y}$
2. residualise $D$ with $Z$ to give $\hat{D}$
3. Fit a linear regression model to the treatment effect, $E[\hat{Y}|\hat{D}]$

Notes:
* a binary treatment/binary outcome seems to be solved with linear regression and residuals are taken as before...?

Partial linear model:
$$
Y = D \theta_0 + g_0(Z) + U, E[U|Z,D]=0
$$
$$
D = m_0(Z) + V, E[V|Z]=0
$$
The uplift $\theta_0$ from the treatment ($D$), is linear in $Y$.

Fully interactive model:
$$
Y = g_0(D, Z) + U, E[U|Z,D]=0
$$
$$
D = m_0(Z) + V, E[V|Z]=0
$$
The uplift from the treatment ($D$), is not linear in $Y$ and is a function of confounders/features.


[ref 1](https://matheusfacure.github.io/python-causality-handbook/22-Debiased-Orthogonal-Machine-Learning.html)
[ref 2](https://towardsdatascience.com/double-machine-learning-for-causal-inference-78e0c6111f9d)

### Propensity scores

Intuition:
Example where we have two actions (pictures of a dog or cat) and we want people to click like or not.
We have logged data from our existing ranker which selects to show dogs 80% of the time and cats 20%.
We can average the number of likes over all the impressions to assess the ranker's value.

Now we want to assess a new proposed ranker which shows dogs 20% of the time and cats 80%.
We need a counterfactual evaluation as the logged data doesn't match the distribution from the proposed ranker.
To assess the value of our proposed ranker we can down scale the reward from dogs and upscale the reward from cats.
We scale to match the expected reward from the distribution of the proposed ranker.
The scale weights come from the propensity scores (importance sampling).
The logged propensity scores for dog are 0.8 and 0.2 for cats. In the proposed ranker they are 0.2 and 0.8.
We use 0.2/0.8 as the weight for dog impression and 0.8/0.2 for the cat impressions.
We multiply the logged rewards by these weights to get an estimate reward for the proposed ranker.

Many cases we do not have the propensity scores from the logging policy. We can fit an offline model to predict the action allocation of the logged data (the direct method). We then use this model to estimate the propensity scores.

### Empirical Bayes
Also known as type-II maximum likelihood.
Calculate a Bayesian prior empircally from the dataset, the same dataset used for observed evidence.

For example in baseball player skill estimation:
The empirical prior would reduce overfitting for individual players by causing shrinkage to the global prior when limited data is available.

P(θ|x) ∝ P(x|θ)P(θ)

P(θ) - prior empirically estimated from all data - league wide skill estimate.
P(x|θ) - likelihood/evidence collected on a single player
P(θ|x) - posterior player estimates

## Statistics


The join probability function denotes is the function mapping over multiple variables, $p_{X,Y}(x,y)$. Where $X, Y$ are random variables and $x, y$ are samples of those random variables.

The conditional probability is the probability of an event given that (conditioned on) another event already happening, $p(x|y)$.

Marginal probabilities denote the probability of an event marginalising over the other variables in the joint distribution, $p(x)$.

For a joint probability function, $p_{X,Y}$, the marginal probability $p(x)$ is:
$$
p_X(x)=\int_y p_{X|Y}(x|y) p_Y(y) dy
$$
To obtain the marginal distribution for $x$ we sum/integrate over the other variables in the joint distribution ($y$).
This is the same as integating the conditional probability of $x$ over the distribution of $y$.


High entropy = uniform distribution

### Likelihood vs probability
Given a conditional probability, P(x|θ), this is interpreted as:
1. a probability - when the parameter, θ, is assumed fixed
2. a likelihood - when the data, x, is assumed fixed

In Bayes rule, P(x|θ), the observed evidence is considered fixed, so this is a likelihood.

### Uncertainty
Epistemic - reduces with more data rows (same features).
Aleatoric - given infinity data rows, the remaining irreducible uncertainty.

## Bandits

We explore over epistemic uncertainty, not aleatoric. For example fitting a beta distribution, the variance shrinks with new data - thus reducing epistemic uncertainty and reducing exploration.