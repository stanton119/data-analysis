# Maths

## Random projection dimensionality reduction
This is useful when the feature space is large.
PCA is computationally expensive as its based on the eigendecomposition (inverse operation).
PCA can be sensitive to outliers as its a linear transform (non robust).
Random projections are fast as we dont calculate the projection matrix based on the data.
We generate it from random gaussians. The number of components for the projection matrix is typically taken from the [Johnson-Lindenstrauss lemma](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_johnson_lindenstrauss_bound.html#sphx-glr-auto-examples-miscellaneous-plot-johnson-lindenstrauss-bound-py).
This states that the pairwise distance between points in the component space is similar to the original space if the number of components is chosen relating to the log of the number of samples.

This is extended through the sparse random projection, which suggests similar results, but with only integer random variables [-1, 0, 1], instead of a gaussian. This is faster for computation again.

[More recent approaches](https://arxiv.org/pdf/2005.00511.pdf) suggest using random projections to first reduce dimensionality and then applying PCA to the result. This combines