# %%
import pandas as pd
import numpy as np
import time
from interpret import show

# %% [markdown]
# # Model/data explanations
# https://github.com/interpretml/interpret#introducing-the-explainable-boosting-machine-ebm

# %%
# Load data
# https://www.kaggle.com/ronitf/heart-disease-uci
"""
age - age in years
sex - (1 = male; 0 = female)
cp - chest pain type
trestbps - resting blood pressure (in mm Hg on admission to the hospital)
chol - serum cholestoral in mg/dl
fbs - (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
restecg - resting electrocardiographic results
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest
slope - the slope of the peak exercise ST segment
ca - number of major vessels (0-3) colored by flourosopy
thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
target - 1 or 0
"""

import numpy as np
np.round(1.5)
df = pd.read_csv(r"heart.csv")
train_cols = df.columns[0:-1]
label = df.columns[-1]

df

# %%
from sklearn.model_selection import train_test_split

np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(
    df[train_cols], df[label], test_size=0.25, shuffle=True
)

# %%
# Explain data - not too useful with binary response
from interpret.data import Marginal

marginal = Marginal().explain_data(X=x_test, y=y_test, name="test data")
show(marginal)


# %%
# Explain models
# Works with anything with fit/predict methods, includes sklearn pipelines
from xgboost import XGBClassifier

t1 = time.time()
xgb = XGBClassifier(seed=0)
xgb.fit(x_train, y_train)
print(time.time() - t1)

xgb.predict(x_test)

# %%
# Show model performance
from interpret.perf import ROC

blackbox_perf = ROC(xgb.predict_proba).explain_perf(
    x_test, y_test, name="XGB ROC"
)
show(blackbox_perf)


# %%
# ## Local predictions
# Interpret ML merges a couple other methods/packages.
# Both methods aim to fit a more simple explainable model around points of interest.
# These approximate explanations.
# %%
# LIME - local interpretable model-agnostic explanations
"""
LIME achieves prediction-level interpretability by approxmiating
the original model with an explanation model locally around that prediction.
Should work with all black box models.
Perturbs the local area and estimates local derivatives to suggest feature importance
https://github.com/marcotcr/lime
"""
from interpret.blackbox import LimeTabular

lime = LimeTabular(predict_fn=xgb.predict_proba, data=x_train, random_state=1)
lime_local = lime.explain_local(x_test[:5], y_test[:5], name="LIME")

show(lime_local)

# %%
# SHAP - SHapley Additive exPlanations
"""
From game theory
Faster than LIME. Fast version for tree based algorithmns.
Graphs are nicer in the original library
https://github.com/slundberg/shap
https://everdark.github.io/k9/notebooks/ml/model_explain/model_explain.nb.html#6_shap
"""
from interpret.blackbox import ShapKernel
import numpy as np

background_val = np.median(x_train, axis=0).reshape(1, -1)
shap = ShapKernel(
    predict_fn=xgb.predict_proba,
    data=background_val,
    feature_names=x_train.columns,
)
shap_local = shap.explain_local(x_test[:5], y_test[:5], name="SHAP")
show(shap_local)


# %%
# Global predictions
# %%
from interpret.blackbox import MorrisSensitivity

sensitivity = MorrisSensitivity(predict_fn=xgb.predict_proba, data=x_train)
sensitivity_global = sensitivity.explain_global(name="Global Sensitivity")

show(sensitivity_global)

# %%
# Partial dependence plots show how each variable or predictor affects the model's predictions
from interpret.blackbox import PartialDependence

pdp = PartialDependence(predict_fn=xgb.predict_proba, data=x_train)
pdp_global = pdp.explain_global(name="Partial Dependence")

show(pdp_global)

# %% [markdown]
# # Explainable boosting machines
# Models that are interpretable from the outset.
#
# GAMs (Generalized additive model) are more flexible than logistic/linear regression, linear sum of non-linear functions of each feature.
# EBM is a GA2M model which aims to include the pairwise interactions between $x$
# GAM:
# $$g(E(y)) = \beta + \sum_i f_i(x_i)$$
# GA2M:
# $$g(E(y)) = \beta + \sum_i f_i(x_i) + \sum_{i\neq j} f_{i,j}(x_i, x_j)$$
# Functions fit via boosting. Splines used to be the norm, but outperformed by boosting with bagging.
#
# To train GAMs via boosting, it fits a decision tree with a single feature and a low learning rate.
# We then train another decision tree with the next feature to fit the residuals over the last.
# Repeat with all features = 1 iteration. Repeat for many iterations.
# We then have a collection of trees on one feature only which we can summarise into a univariate relationship.
# This is where the iterpretability comes in.  
# Video: https://www.youtube.com/watch?v=MREiHgHgl0k
# 
# Galaxy zoo paper compares against EBM, https://arxiv.org/pdf/1905.07424.pdf.
#
# "EBM aim to predict a target variable based on tabular features by separating the impact of those features into single (or, optionally, pairwise) effects on the target variable. They are a specific implementation of Generalised Additive Models (GAM, Hastie & Tibshirani 1990). GAM are of the form:
#
# $g(y) = f_1(x_1) + ... + f_n(x_n)$
#
# For EBM, each $f_i$ is learned using gradient boosting with bagging of shallow regression trees.
# They aim to answer the question ‘What is the effect on the target variable of this particular feature alone?'"

# %%
# slower than xgboost, have built in cv, performance not comparible
from interpret.glassbox import ExplainableBoostingClassifier

t1 = time.time()
ebm = ExplainableBoostingClassifier(random_state=0, interactions=1)
ebm.fit(x_train, y_train)
print(time.time() - t1)

ebm.predict(x_test)
y_test

# %%
# Performance is claimed to be better on average than XGBoost
# https://github.com/interpretml/interpret#introducing-the-explainable-boosting-machine-ebm
from interpret.perf import ROC

glassbox_perf = ROC(ebm.predict_proba).explain_perf(x_test, y_test, name="EBM")
show(glassbox_perf)


# %%
# Explain global feature trends
from interpret import show

ebm_global = ebm.explain_global()
show(ebm_global)

# %%
# Explain individual predictions, prob of class shown
ebm_local = ebm.explain_local(x_test, y_test)
show(ebm_local)


# %%
# Show all in dashboard
show(
    [
        marginal,
        lime_local,
        shap_local,
        glassbox_perf,
        blackbox_perf,
        sensitivity_global,
        pdp_global,
    ]
)


# %%
# Other models
from interpret.glassbox import LogisticRegression, ClassificationTree

t1 = time.time()
log_reg = LogisticRegression(random_state=0)
log_reg.fit(x_train, y_train)
print(time.time() - t1)

log_reg.predict(x_test)
y_test
# Explain global feature trends
from interpret import show

log_reg_global = log_reg.explain_global()
show(log_reg_global)


t1 = time.time()
class_tree = ClassificationTree(random_state=0)
class_tree.fit(x_train, y_train)
print(time.time() - t1)

class_tree.predict(x_test)
y_test
# Explain global feature trends
from interpret import show

class_tree_global = class_tree.explain_global()
show(class_tree_global)

# %% [markdown]
# ## Conclusions
# * The explanable boosting machine seems useful to bridge performance and interpretability
# * Authors suggest they should supersede boosted trees, random forests, linear/logistic regression
# * The dashboard seems a nice to have. Doesnt really add anything too useful for us. Maybe for presenting results
# * Converting to the built in explainable implementations doesnt seem worth learning.
# * Worth using shap by itself though, probably over LIME.