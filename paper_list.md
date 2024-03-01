# Paper list
Summaries of random papers 

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

### Decoupling magnitude and phase estimation with deep ResUNet for music source separation
https://arxiv.org/pdf/2109.05418.pdf
*   Network designed for source separation
    *   Generates masks for various instruments etc. which are then applied to the original sound STFT
    *   Applied to dataset - MUSDB18
*   novelty:
    *   Uses complex masks rather than magnitude only based masks to decouple masks into magnitude and phase
    *   Phase representation improves on previous methods' degradation
    *   More appropriately weighted masks which do not have to sum to 1
    *   Deeper network of residual UNet layers

### A Contextual-Bandit Approach to Personalized News Article Recommendation
https://arxiv.org/abs/1003.0146
*   2012
*   News articles are inpractical for colab filters as items to recommend are frequently changing
    *   Cold start problem is big
*   maximises click through rate by select optimal stories to recommend
*   Uses info on user and article
*   Suggests how the methods can be tested offline with random data

### A Survey of Uncertainty in Deep Neural Networks
https://arxiv.org/pdf/2107.03342.pdf
* 2022
* Summary paper discussion methods to capture uncertainty in neural networks.
* Single Deterministic Methods
  * Internal Uncertainty Quantification Approaches - using a single network to predict distribution parameters.
    * Classification problems have commonly been modelled with Dirichlet distributions over the binary outputs.
    * Dirichlet (multivariate beta distribution) is the conjugate prior of the multinomial distribution which corresponds to a multi class classification problem.
  * External - using a separate network to model the uncertainty alone.
  * These methods are fast at inference time due to a single forward pass used, though can be slower during training due to more complex loss functions.
* Bayesian Networks
  * Learns distributions over the network parameters.
  * The network outputs are found by marginalising over the network parameters, however this is typically intractable.
  * Inference is therefore costly as we typically sample from the parameters to form a distribution over the network outputs.
  * Practical solutions include varitional inference, sampling approaches and the Laplace approximation.
  * Variational inference forces the posterior parameter distributions to follow a family of tractable solutions.
    * This includes ELBO approaches and MC drop out
  * Sampling includes MCMC.
  * Laplace approximation assumes the loss surface is uni-modal and approximates it with a normal distribution.
    * The parameter log posterior distribution is represented by the Hessian, which is complex to compute for NNs and typically requires approximate solutions.
* Ensemble methods
  * Using multiple models can converge to different local optima, therefore we can more easily get multi-modal output distributions.
  * Need to maximise the variety across single networks to maximise generalisation.
    * Methods include random initialisation, random data shuffling, standard practices (bagging, boosting), data augmentation and ensembling across architectures.
  * Random initialisation is shown as a good solution. Bagging has been found to degrade uncertainty estimation.
  * Ensembles found to be more reliable than MC dropout.
  * Ensembles are computationally complex. Pruning or distillation of the ensemble members can help.
* Test time augmentation
  * This uses data augmentation on the input data (mostly applied to image data).
  * Each version is passed through a single network and the results are collected to form a probabilistic output.
* Measurement - the paper looks at methods to quantify performance which I did not read in detail.

### Evidential Deep Learning to Quantify Classification Uncertainty
https://arxiv.org/pdf/1806.01768.pdf
* 2018
* Classification models only with softmax are over confident in predictions when uncertainty is high.
* Proposed to model the NN output as a Dirichlet distribution (multivariate beta distribution).
* The parameters of the distribution represent the evidence in each output class.
* The network weights are trained via the likelihood.
* A classifier network maximises the loglikelihood of a multinomial distribution over the class probabilities.
* The multinomial probabilities come from the Dirichlet prior distribution.
* The multinomial class probabilities are marginalised out to give the likelihood from the Dirichlet evidence parameters.
* This is the same approach as empirical Bayes and Type II maximium Likelihood.
* Probability simplex = a unit vector (sums to one) where each component is between 0 and 1.
* They add a KL divergence term to a uniform distribution to penalise divergences for uncertain predictions that do not contribute towards the data fit.
* Results
  * It was tested comparing against other uncertainty methods by training the same network architecture each time.
  * Uses MNIST and CIFAR10. The first 5 from CIFAR are used for training, the last 5 used for uncertainty quanitifcation.
  * Predictive performance is similar to other methods like MC dropout.
  * Uncertainty estimates are improved as measured in entropy on dummy MNIST data.
* Code: https://muratsensoy.github.io/uncertainty.html

### The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits
https://arxiv.org/abs/2402.17764v1
* Summary
  * Quantises LLM to signed binary (ternary) values. This makes multiplication become additions.
  * Reports similar performance to full precision LLMs for the same model size and training tokens.
* Background
  * Most quantisation efforts are applied post training, but is suboptimal.
  * Previous 1-bit BitNet was using only -1 and 1 as weights.
    * https://arxiv.org/abs/2310.11453
  * This paper has -1, 0, 1 for 1.58 bits per weight. Same benefits but improved performance.
* Method
  * The model is trained from scratch.
  * The weights in a layer are quantized by scaling by the average absolute value, rounding to integers and clipping to [-1,1]
* Results
  * Perplexity (exponential of log likelihood of seing the next token)
  * Performances better than FP16 model at large sizes but worse on small sizes.
  * Larger models benefit significantly from ternary weights.
* Questions
  * Bias terms also ternary? - has no bias terms
  * Hardware implications?
  * Performance gains at same energy consumption?
  * How to train effectively? Gradient descent needs to be quantised?
  * Why do we not have FP16 additions if the original embeddings are floating points? Are they quantised as well - wouldn't make much sense unless the embedding space has very high dimension?