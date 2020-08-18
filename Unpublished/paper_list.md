# Paper list

Curveball - https://arxiv.org/abs/1805.08095
* Small steps and giant leaps: Minimal Newton solvers for Deep Learning
* Newton optimisation for large networks
* Requires Hessian which is hard to store/invert/noisy
* Uses stochastic gradient descent to solve Newton step
* Doesn't require the need to store/invert
* Computationally equivalent to SGD momentum (2 vectors)

The Power Of Deeper Networks For Expressing Natural Functions - https://arxiv.org/pdf/1705.05502.pdf
* The number of neurons m required to approximate natural classes of multivariate polynomials of n variables grows only linearly with n for deep neural networks, but grows exponentially when merely a single hidden layer is allowed

Universal Approximation with Deep Narrow Networks - https://arxiv.org/pdf/1905.08539.pdf
* The classical Universal Approximation Theorem holds for neural networks of arbitrary width and bounded depth
* Here we consider the natural ‘dual’ scenario for networks of bounded width and arbitrary depth

Towards Understanding Generalization of Deep Learning: Perspective of Loss Landscapes - https://arxiv.org/pdf/1706.10239.pdf
* Neural networks generalize well, even with much more parameters than the number of training samples
* Explains that good minima on the loss landscape have flat hessians, and also a large basin of attraction
* Random starting points nearly always converge to a good minima which generalizes