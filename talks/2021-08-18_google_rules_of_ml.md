# Google Rules of ML

https://developers.google.com/machine-learning/guides/rules-of-ml/

Summary:
*   Build simple heuristics over ML models straight away
*   ML over complex heuristics
*   Build out a simple model first
*   Build simple metrics/objectives
*   Feature engineering


Useful hints:
*   4 - keep model simple, production infrastructure
*   5 - make it testable
*   7 - use existing heuristics
*   8 - models go stale
*   12 - dont overthink initial objective function
*   13 - simple observable metric first
*   14 - start with interpretable models
*   16 - plan on iterating and launching
*   17 - start with external features not learned ones
*   21 - number of features to learn relates to data size
*   24 - difference between production and proposed model
*   25 - end result is more important than model prediction accuracy
*   26 - features for patterns seen in model errors
*   29 - train vs serve feature difference
*   32 - reuse code train vs serve
*   33 - train/test split
*   34 - dont introduce sampling bias to hold out datasets
*   39 - launch decisions are not always ML metrics
*   40 - keep ensembles simple
*   41 - add new info rather than improve existing