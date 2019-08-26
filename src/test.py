import numpy as np
import pandas as pd
import holoviews as hv
import matplotlib.pyplot as plt
from holoviews import opts
hv.extension('plotly')

#%%

n = 1000
m = 5
temp: np.ndarray = np.random.randint(low=0, high=10, size=(n, m))

print(temp.shape)


#%%
n = 10000
m = 5
df = pd.DataFrame(np.concatenate((np.random.randint(0, high=5, size=(n,2)),
                                 np.random.randn(n,m-2)), 
                                 axis=1),
                  columns=[str(x) for x in range(m)])

df
plt.plot(df['0'], df['3'], '.')
plt.show()

#%%
# scatter = hv.Scatter(df, '0', '1')
# hv.extension('plotly')
hv.extension('bokeh')
# hv.extension('matplotlib')


# vdims = [('measles', 'Measles Incidence'), ('pertussis', 'Pertussis Incidence')]
ds = hv.Dataset(df, ['0', '1']) #, ['Year', 'State'], vdims)

layout = (ds.to(hv.Scatter, '2', '3') + ds.to(hv.Scatter, '2', '4')).cols(1)
layout.opts(
    opts.Scatter(color='red', tools=['hover'])).cols(1)

# scatter = hv.Scatter(df, kdims=['0'])
# scatter


#%%
df = pd.DataFrame([10, 5],
                  columns=['col1'])

type(df['col1'])

#%%
