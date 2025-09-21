# Recommender systems

## Literature reviews
1. Conferences
   1. Recsys, SIGIR


## Notes
*   Collaborative filtering
    *   Filter items to predict user preference
    *   Based on information from other users
    *   Overview
        *   User rates some items in a system
        *   Find similar users based on those items
        *   Recommend items rated highly by other users that has not been seen yet
    *   Similar users
        *   cosine similarity commonly used
    *   Recommended items scores
        *   Similarity to other users * their scores
    *   Issues
        *   Scales badly
        *   Sparsity on ratings makes this hard
        *   Users that dont fit into standard clusters can be difficult
        *   Populatity bias
        *   Cold start - new items cant be recommended
    *   ML approaches typically utilise dimensionally reduction on the item space
*   Content based filtering
    *   Describe each item in terms of attributes
    *   Dont rely solely on user ratings
    *   Lower cold start problem
    *   Matrix factorisation
        *   SVG estimates
            *   http://nicolas-hug.com/blog/matrix_facto_1
    *   Cosine similarity
        *   Spotify example
            *   Get features per item/song etc.
            *   Given a playlist, find average features of playlist by averaging songs
            *   Find lowest cosine similarity with other songs = recommended songs
*   User centric models - collab filtering etc
    *   Optimises single objective
*   Joint stakeholders - sellers and buyers - marketplaces
    *   Artists/users
    *   Multi objectives


*   Spotify
    *   http://rishabhmehrotra.com
        *   https://www.youtube.com/watch?v=KoMKgNeUX4k
        *   Need to optimise fairness to help artists who are more niche
        *   Relevance - only what the user wants to see
        *   Optimising multi objectives with constraints
            *   Dont let relevance drop too much to improve fairness
        *   User specific thresholds
        *   Can improve fairness without reduce relevance too much, elbow in graph
        *   Uses contextual bandits for recommendations
        *   Optimising for multiple metrics generally performs better on each of those metrics than optimising for just that one metric
    *   https://www.youtube.com/watch?v=HKW_v0xLHH4
        *   2017 talk
        *   Song embeddings
            *   Collaborative filter based
            *   Repesenting songs in a vector form, similar to NLP word embeddings
            *   Allows song similarity scores
            *   Allows math operations to go from song1 + song2 - song3 = song4
        *   Features from sound
            *   Help with cold start issues
            *   Built CNN on spectrogram of raw audio
                *   CNN + pooling -> dense layers -> dense layers
            *   Output was latent space vector
                *   From collaborative filtering labels


*   automatic playlist continuation
    *   
*   Spotify sequential skip prediction challenge
*   https://research.atspotify.com/datasets/


*   Spotify questions
    *   Niche artists more uncertainty on recommendations
    *   How do you simulate the environment to test various recommender systems?


*   Automatic playlist continuation
    *   Methods
        *   Naive Collab filtering
            *   Each playlist is a user
            *   One hot encode each track
            *   Assign 1 to each each track for a playlist
            *   Find cosine similarity for all playlists
            *   Weighted sum of tracks not seen in the current playlist
    *   Papers
        *   Automatic playlist continuation using a hybrid recommender system combining features from text and audio
            *   https://arxiv.org/abs/1901.00450
            *   Ferraro, 2019

### Thoughts
1. When ranking millions of items we use
   1. a candidate selection model to fast filter items down to a 100 or so
      1. Inner product based is common for scalability and speed. Aims to have high recall to capture anything that is relevant to the user.
   2. a ranker which runs more detailed scoring
      1. Aims to have high precision to focus on high quality items.
2. Recommender systems vs bandits?
   1. Any recommender system becomes a bandit by introducing exploration
   2. This could be by adding an uncertainty layer and then sample from it. Or just an epsilon greedy or UCB heuristic approach.
   3. Bandits are used to explore around epistemic uncertainty
      1. Epistemic uncertainty relates to areas in the data space which are unexplored
      2. Bandits select actions with higher epistemic uncertainty
      3. Epsilon greedy uniformly collects data, regardless of where epistemic uncertainty is high. Thompson sampling samples from the action reward posterior distributions which optimises rewards in the presence of epistemic uncertainty.
   4. Recommender systems typically train on very large datasets where we assume epistemic uncertainty is lower
      1. However many systems have long tailed item distributions, cold start issues etc., meaning exploration is still necessary
   5. Propensity scores
      1. Needed for accurate OPE
      2. Not always required for training models - if confounding features (those that relate to the selection bias) are present, then no need for inverse propensity scoring (direct modelling causal approach)
      3. Recommender systems typically dont explore to create accurate propensities. Therefore they are evaluated with NCDG/precision/recall
      4. Well defined exploration strategies enable accurate OPE which would be preferred if possible.
3. 2025 themes
   1. Long sequences
      1. How long do sequences have to be to be a problem at current?
      2. Long in terms of time or number of items?
      3. What if users have limited sequence lengths?
   2. LLM augmentation
      1. Features that come from LLMs
   3. Diffusion models
      1. To model user/item interaction matrix with diffusion.
   4. Semantic IDs
      1. Instead of items having random ordinal IDs, use a hieracrchical ID setup
   5. Generative models

## Main concepts
1.  Implicit vs explicit feedback
    1.  Explicit = users giving direct ratings/feedback/relevance labels on an item. Harder to gather, less biased.
    2.  Implicit = assumed relevance from user actions like clicks/purchases etc. to inform positive relevance. Assumes non acted items are negatives/non-positives. Easier to gather data, adds bias.

## Bandit models
*   Multi armed bandits
    *   https://www.youtube.com/watch?v=e3L4VocZnnQ&list=WL&index=22
    *   Compromise in optimisation problems between exploration and exploitation
    *   Exploration to find parameters from our unknown environment
    *   Exploitation to maximise your current knowledge
    *   Regret = difference in objective metric between perfect score given perfect information, and another method
        *   Hard to observe in real datasets
    *   Some benchmarks
        *   exploration only - do not use current knowledge, act randomly
        *   exploit only
        *   epsilon greedy
            *   e.g. 10% chance of acting randomly and exploring, 90% chance of exploiting
    *   Example
        *   Restaurants with distribution
        *   Visit 300 times, maximise happiness
        *   Implement epsilon with varying % of exploration
        *   Find optimal % param
        *   Should you push the exploration to the start? e.g. first % of iterations?
    *   Simulated annealing - hyper parameter optimiser with exploration vs exploitation
*   Experimentation setup
    *   Offline evaluation is tricky as environment needs to respond to your model output
    *   Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms
        *   https://arxiv.org/pdf/1003.5956.pdf
        *   Li, 2012
        *   **Replay** method for offline evaluation of solutions
        *   Data driven, avoids modelling bias
        *   Data used will be from some production system which builds in it's bias
        *   Removes data rows where the recommendations of the proposed model are incompatible with the dataset
            *   Does this create an upper limit on training performance if the dataset is too sub optimal?
        *   Requires a large training set
        *   Need metric that's independent of data size?
    *   https://jamesrledoux.com/algorithms/offline-bandit-evaluation/
        *   Batch updating is more time effectively - update after n rows
        *   Slate recommendations - commonly used
            *   We only have data for the recommendations that were actually presented
            *   Our bandit models give a recommendation
                *   if that recommendation was never presented to the user in the training set we cant learn from it
            *   We need to keep rows that conform to our model output
            *   Removes much of the data
            *   Recommend multiple items - keeps more data
                *   Higher chance that one of those recommendations was seen by the user
*   Contextual bandits
    *   https://towardsdatascience.com/contextual-bandits-and-reinforcement-learning-6bdfeaece72a
    *   Multi armed bandits make decisions without any knowledge of the environment
    *   E.g. instead of randomly assigning to a user, based the decision based on their preferences

*   Thompson sampling
    *   Solution to multi armed bandit problems
    *   Each option has a prior distribution of its reward
        *   Can be uninformed prior
    *   Sample from the prior to get a sampled reward for each option
    *   Order and e.g. pick the top
    *   Update the prior with data to get a posterior distribution for the reward
    *   Repeat until posterior distributions have converged

*   Datasets
    *   https://grouplens.org/datasets/movielens/25m/
    *   https://research.atspotify.com/datasets/
*   

## Colaborative filters
Data format is typically taken in the format of user/item interactions:
```
User ID	Item ID	Time
User 1	Item 1	2015/06/20T10:00:00
User 1	Item 1	2015/06/28T11:00:00
User 1	Item 2	2015/08/28T11:01:00
User 1	Item 2	2015/08/28T12:00:01
```

1.  Singular value decomposition SVD
    1.  Given a user/item rating matrix (R), decompose in to U and V. Where U/V are user/item latent embeddings of dimension L. L << R.
    2.  U/V dimensions are orthogonal.
2.  Neural colab filters
    1.  Combines two avenues which are concatenated and projected to a scalar:
        1.  Inner product of user and item embeddings and
        2.  Given user and item embeddings, concatenate them and push through an MLP
3.  SAR - Smart Adaptive Recommendations
    1.  Based on implicit feedback
    2.  https://github.com/Microsoft/Product-Recommendations/blob/master/doc/sar.md
4.  DCN - deep and cross network (+v2)
    1.  Cross network - explicit feature interactions from cross multiplications
        1.  Features are multiplied by weight matrices that are multipled by the features again
        2.  Weight matrices are decomposed into e.g U/V matrices for efficiency
    2.  Deep network - typical MLP
5.  
