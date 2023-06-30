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