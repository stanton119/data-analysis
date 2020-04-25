# %% [markdown]
# # Heart Disease Data Exploration
#

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.style.use("seaborn-whitegrid")


# %%
# Fetch data
dir_path = os.path.dirname(os.path.realpath(__file__))
heart_data = pd.read_csv(os.path.join(dir_path, "data", "heart.csv"))
# data retrieved from: https://www.kaggle.com/ronitf/heart-disease-uci
print(heart_data.shape)
heart_data.head()

# %% Clean data
# No missing/inf values
print("Missing:\n", heart_data.isna().sum(), "\n")
print("Inf:\n", (np.abs(heart_data) == np.inf).sum(), "\n")

heart_data.describe()



# %% Correlation plots
sns.pairplot(heart_data)
# plt.savefig('TFLCycles/images/pairplot.png')
plt.show()

# %% Interactive plots
# hvplot on dataframes
# panels/holoviews for interactivity

# %%
X_train = heart_data.drop(columns='target')
y_train = heart_data['target']

# %% Logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model = model.fit(X=X_train, y=y_train)
model.predict(X_train).mean()
model.coef_
X_train.columns
model.intercept_
model.get_params()

# %% Explainable gbm
from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression
from interpret import show


ebm = ExplainableBoostingClassifier()
ebm.fit(X=X_train, y=y_train)

ebm_global = ebm.explain_global(name='EBM')
show(ebm_global)

# %%
log_model = LogisticRegression()
log_model.fit(X=X_train, y=y_train)
log_global = log_model.explain_global(name='LogReg')
show(log_global)

show([ebm_global, log_global], share_tables=True)


# %%
from interpret.data import ClassHistogram
hist = ClassHistogram().explain_data(X_train, y_train, name = 'Train Data')
show(hist)


ebm_local = ebm.explain_local(X_train[:5], y_train[:5], name='EBM')
show(ebm_local)

# %%
from interpret.perf import ROC

blackbox_perf = ROC(model.predict_proba).explain_perf(X_train, y_train, name='SkLearnLog')
show(blackbox_perf)
blackbox_ebm_perf = ROC(ebm.predict_proba).explain_perf(X_train, y_train, name='EBM')
show(blackbox_ebm_perf)

# %%
from interpret.blackbox import LimeTabular

#Blackbox explainers need a predict function, and optionally a dataset
lime = LimeTabular(predict_fn=model.predict_proba, data=X_train, random_state=1)

#Pick the instances to explain, optionally pass in labels if you have them
lime_local = lime.explain_local(X_train[:5], y_train[:5], name='LIME')

show(lime_local)

from interpret.blackbox import ShapKernel
import numpy as np

background_val = np.median(X_train, axis=0).reshape(1, -1)
shap = ShapKernel(predict_fn=model.predict_proba, data=background_val, feature_names=X_train.columns)
shap_local = shap.explain_local(X_train[:5], y_train[:5], name='SHAP')
show(shap_local)

from interpret.blackbox import MorrisSensitivity

sensitivity = MorrisSensitivity(predict_fn=model.predict_proba, data=X_train)
sensitivity_global = sensitivity.explain_global(name="Global Sensitivity")

show(sensitivity_global)

from interpret.blackbox import PartialDependence

pdp = PartialDependence(predict_fn=model.predict_proba, data=X_train)
pdp_global = pdp.explain_global(name='Partial Dependence')

show(pdp_global)

show([blackbox_perf, blackbox_ebm_perf, lime_local, shap_local, sensitivity_global, pdp_global])


# %% Run via CV
"""
Not much data, many features
"""
# %% Compare to elastic net
