# Classification models

*   Binary classifier model
    *   gives a probability of True
    *   Threshold used to say whether the output is true or not, e.g. P(x)>50% = True
*   [Precision recall](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/)
    *   High precision
        *   Precision = TruePositives / (TruePositives + FalsePositives)
        *   for all true predictions, there is a high chance that the actual class is true
        *   we are more happy to classify something incorrectly as negative, rather than incorrectly as positive
        *   low chance of false positive, higher chance of false negative
    *   High recall
        *   Recall = TruePositives / (TruePositives + FalseNegatives)
        *   For all positive cases we want to ensure as many as possible are predicted as true by the model
        *   Dont want to miss out anything that could be positive
        *   Low chance of false negative, higher chance of false positive
    *   ROC (receiver operator curve)
        *   Plot the true positive rate against (y) the false positive rate (x)
            *   TruePositiveRate = TruePositives / (TruePositives + False Negatives)
            *   FalsePositiveRate = FalsePositives / (FalsePositives + TrueNegatives)
        *   Changing the threshold on the probability of when to classify as true or not, gives a range of values
        *   Area under the curve (AUC) represents the skill of the model
        *   TruePositiveRate vs FalsePositiveRate for varying classification thresholds
        *   Diagonal lines = random
        *   Higher = good, lower = bad
        *   Not biased towards majority classes
    *   AUC (Area under curve)
        *   Sum of area under ROC graph
        *   0.5 = random, 1 = good, 0 = bad
        *   Useful for comparing classifiers on unbalanced problems 
            *  when we have sufficient samples
            *  otherwise we have a high variance on those samples from the minority class
    *   Precision recall plot
        *   