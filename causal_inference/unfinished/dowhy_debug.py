# %%

import pandas as pd
import numpy as np
import dowhy
from IPython.display import Image, display


# %%

n_samples = 1000
n_features = 3
treatment_binary = True

rand = np.random.default_rng(0)
x = rand.normal(loc=rand.normal(size=n_features), size=(n_samples, n_features))


if treatment_binary:
    t = rand.binomial(n=1, p=0.5, size=(n_samples, 1))
else:
    t = rand.normal(size=(n_samples, 1))
bias = rand.normal()
weights = rand.normal(size=(n_features + 1, 1))
y = bias + np.dot(np.concatenate([t, x], axis=1), weights) + rand.normal()
# collider variable
x[:, [2]] = y + 0.2 * t + 2*rand.normal(size=(n_samples, 1))

x_cols = [f"x{idx+1}" for idx in range(3)]
t_col = "t"
y_col = "y"
df = pd.DataFrame(x, columns=x_cols)
df[t_col] = t
df[y_col] = y
df = df[[y_col, t_col] + x_cols]


print("\n")
print("True weights:")
print(bias, weights)

# %%
causal_graph = """
digraph {
    t -> y;
    x1 -> y;
    x2 -> y;
    t -> x3;
    y -> x3;
}
"""

causal_model = dowhy.CausalModel(
    data=df, graph=causal_graph.replace("\n", " "), treatment=t_col, outcome=y_col
)
causal_model.view_model()
display(Image(filename="causal_model.png"))

# %%
# Identify the causal effect
estimands = causal_model.identify_effect()

# Causal Effect Estimation
estimate = causal_model.estimate_effect(
    estimands, method_name="backdoor.linear_regression"
)
print(estimate)
# %%
import sklearn.linear_model


model = sklearn.linear_model.LinearRegression()
model.fit(df[[t_col]], df[y_col])
print(model.coef_)
