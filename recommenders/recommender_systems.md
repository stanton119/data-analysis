# Recommender systems

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
                *   We need to keep rows that conform to our model output
                *   Removes much of the data
                *   Recommend multiple items - keeps more data
            *   


*   Datasets
    *   https://grouplens.org/datasets/movielens/25m/
    *   https://research.atspotify.com/datasets/