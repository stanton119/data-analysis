# Paper list

### Curveball
https://arxiv.org/abs/1805.08095
* Small steps and giant leaps: Minimal Newton solvers for Deep Learning
* Newton optimisation for large networks
* Requires Hessian which is hard to store/invert/noisy
* Uses stochastic gradient descent to solve Newton step
* Doesn't require the need to store/invert
* Computationally equivalent to SGD momentum (2 vectors)

### The Power Of Deeper Networks For Expressing Natural Functions
https://arxiv.org/pdf/1705.05502.pdf
* The number of neurons m required to approximate natural classes of multivariate polynomials of n variables grows only linearly with n for deep neural networks, but grows exponentially when merely a single hidden layer is allowed

### Universal Approximation with Deep Narrow Networks
https://arxiv.org/pdf/1905.08539.pdf
* The classical Universal Approximation Theorem holds for neural networks of arbitrary width and bounded depth
* Here we consider the natural ‘dual’ scenario for networks of bounded width and arbitrary depth

### Towards Understanding Generalization of Deep Learning: Perspective of Loss Landscapes
https://arxiv.org/pdf/1706.10239.pdf
* Neural networks generalize well, even with much more parameters than the number of training samples
* Explains that good minima on the loss landscape have flat hessians, and also a large basin of attraction
* Random starting points nearly always converge to a good minima which generalizes

### Understanding Deep Learning Requires Rethinking Generalization
https://arxiv.org/pdf/1611.03530.pdf
* Experiments establish that state-of-the-art convolutional networks for image classification trained with stochastic gradient methods easily fit a random labeling of the training data
  * The effective capacity of neural networks is sufficient for memorizing the entire data set.
* This phenomenon is qualitatively unaffected by explicit regularization, and occurs even if we replace the true images by completely unstructured random noise
  * weight decay, dropout, and data augmentation, do not adequately explain the generalization error of neural networks
  * Explicit regularization may improve generalization performance, but is neither necessary nor by itself sufficient for controlling generalization error
* We corroborate these experimental findings with a theoretical construction showing that simple depth two neural networks already have perfect finite sample expressivity as soon as the number of parameters exceeds the number of data points as it usually does in practice.
* Shows that SGD acts as an implicit regularizer

### Regularization for Deep Learning: A Taxonomy
https://arxiv.org/pdf/1710.10686.pdf
* Regularization via data
  * Training set augementation
  * Adding noise to training samples
  * Dropout
  * Batch normalisation
* Regularization via the network architecture
  * Limiting the search space can find better solutions
  * Weight sharing (e.g. conv nets)
  * Activation functions
  * Noisy models
  * Dilated convolution
  * Max pooling layers
  * Dropout
* Regularization via the error function
* Regularization via the regularization term
* Regularization via optimization


### On The Variance Of The Adaptive Learning Rate And Beyond
https://arxiv.org/pdf/1908.03265.pdf
* Adam starts with a too high learning rate
  * When it estimates the momentum/gradients the high learning rate means there is a high variance in the distribution of gradients and they are poorly estimated
* Adam with a low initial learning rate gets a better estimate of the gradients

### Galaxy Zoo: Classifying Galaxies with Crowdsourcing and Active Learning
https://blog.tensorflow.org/2020/05/galaxy-zoo-classifying-galaxies-with-crowdsourcing-and-active-learning.html
* At train time, dropout reduces overfitting by “approximately combining exponentially many different neural network architectures efficiently” (Srivastava 2014).
* This approximates the Bayesian approach of treating the network weights as random variables to be marginalised over.
* By also applying dropout at test time, we can exploit this idea of approximating many models to also make Bayesian predictions (Gal 2016).

### Class-Balanced Loss Based on Effective Number of Samples
https://arxiv.org/pdf/1901.05555.pdf
* With class imbalanced problems re-weighting by inverse class frequency is common
* This causes slow training and other problems by not effectivley learn on the most represented classes
* As number of samples grows, the benefit of additional samples is diminished
* Suggests finding an effective number of samples of the most represented classes and use that to re-weight
* Uses the same re-weighting function across all samples, treats as a hyperparameter and improves overall performance

### Implicit Gradient Regularization
https://arxiv.org/abs/2009.11162
* SGD with a large learning rate has a better change of reaching flatter minima
* This is from implicit regularisation from gradient descent step which discretise the continuous loss function gradient
* Forms an explicit regularisation which helps formulate this

### Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
https://arxiv.org/abs/2006.16236
* Transformers combine layer outputs from multiple inputs
* The weights comes from a softmax which is a function of nearby inputs and trained coefficients
* Due to the quadratic term, transformers are slow to compute
* By approximating the kernel and not using the exponential kernel the quadratic can become linear
* Speed up in performance
* Same method used to show transformers have an equivalent representation by RNNs

### AR-Net: A Simple Auto-Regressive Neural Network For Time-Series
https://arxiv.org/abs/1911.12436
* AR models scale quadratically with the number of lags we consider
* This paper fits AR models with SGD and achieves lienar scaling
* Enables very long lag AR models
* AR coefficients match classical AR models
* Coefficients are regularised to introduce sparseness
* No need for seasonal decomposition models if we can use AR?
