# Paper list
Summaries of random papers. Usually not updated...

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

### Deep Bayesian Bandits: Exploring in Online Personalized Recommendations
https://arxiv.org/abs/2008.00727
* Summary
  * Estimates posterior distribution from a deep neural network for use in bandit applications
* Background
  * Dropout is computationally expensive to train and predict
  * Bootstrapping NNs - multiple networks, with a sample trained for each is computationally expensive
* Method
  * Uses dropout in the 2nd last layer only
  * Has multiple output heads for the same output prediction
  * Dropout applies a Bernoulli mask which approximates bootstrap sampling
  * Can use a single pass of the shared layers and parallel pass of the final multiple output heads for approximate posterior inference
  * Uses a fully connected NN to estimate CTR
* Results
  * Tests by doing off-policy and online evaluation
  * Dropout based models perform worse likely due to increased time to converge during training
  * Did not observe significant performance benefits but has lower computation complexity

### ADAM: A Method For Stochastic Optimization [2015]
https://arxiv.org/pdf/1412.6980.pdf
Read 2024/03
* Summary
  * optimiser for parameters that can handle non-stationary gradients, sparse gradients and a stochastic optimisation function.
  * Based on adaptive estimates (exponentially average) of the gradient first and second order moments ($E\{g\}$, $E\{g^2\}$).
  * The gradient moment estimates are vectors and we have a single learning for all parameters.
* Algorithm
  * At every time step, $t$,  (batch update) we calculate gradients, $g_t$, of $f(\theta_t)$.
  * Where $\theta$ are our optimisation parameters (NN model weights).
  * We update our estimate of first and second order moments of the gradients as an exponential average:
    * $m_t = \beta_1 * m_{t-1} + (1-\beta_1) * g_t$
    * $v_t = \beta_2 * v_{t-1} + (1-\beta_2) * g^2_t$
    * Typical values for $\beta_1=0.9$ and $\beta_2=0.999$, which suggests slow/stable updates to the gradient first moment estimate, and very slow updates to the second order moment.
  * Then we update our parameters:
    * $\theta_t = \theta_{t-1} - \alpha * m_t/(\sqrt{v_t}+\epsilon)$,
    * where $\alpha$ is the learning rate and $\epsilon$ is a parameter for tuning non-stationarity effects.
    * When the second order moment estimate is high (high uncertainty) then the updates are small.
    * The typical max step size is approx bounded by $\alpha$.
  * The gradients are assumed to be initialised to 0 and therefore biased towards 0.
    * So a bias correction term is applied.
    * So the expected values are updated in the above as:
    * $m_t = m_t/(1-\beta^t_1)$
    * $v_t = v_t/(1-\beta^t_2)$
    * The higher the $\beta$ values the more time we average over and the lower the bias correction.
* Convergence analysis
  * Shows regret bound of $O(\sqrt{T})$
  * Using a learning rate decay of $\alpha_t=\alpha/\sqrt{t}$ is common.
* Results
  * Performs similarly to SGD with momentum on dense MNIST
  * Performas similarly to AdaGrad on sparse problems (best of both worlds)

### The LambdaLoss Framework for Ranking Metric Optimization [2018]
https://dl.acm.org/doi/10.1145/3269206.3271784
Read 2024/03
* Summary
  * Formulates a general framework for optimise learn to rank problems

### Unbiased Learning to Rank Meets Reality: Lessons from Baidu’s Large-Scale Search Dataset [2024]
https://dl.acm.org/doi/pdf/10.1145/3626772.3657892
Read 2025/01
* Summary
  * Applying unbiased learn-to-rank methods to remove biases such as position bias do not improve ranking performance on the Baidu ULTR dataset.
  * ULTR improved the click model loss function but this didn't translate into ranking improvements.

### A Semi-Personalized System for User Cold Start Recommendation on Music Streaming Apps
http://arxiv.org/pdf/2106.03819
Read 2025/02
* Summary
  * Deezer - recommending songs to stream.
  * In a colaborative filter recommender system, cold start users do not meaningful embeddings.
  * Using a dataset from warm start users, predict their embeddings given only demograpic and 1 day of streaming data.
  * Use this model to predict embeddingss for cold start users from the same limited data.
  * Use these predicted embeddings in a k-NN full personalised approach
  * Separately they make semi-personalised approach.
  * They cluster warm users embeddings into 1000 centroids.
  * For each cluster they find the most popular songs.
  * The cold start users predicted embedding is matched to the closest centroid.
  * They are recommended the clusters popular songs.
* Results
  * Results suggest semi personalised approach is superior to all other approaches tried.
  * The results are generally quite poor, so maybe some doubt on the setup.
