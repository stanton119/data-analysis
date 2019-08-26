#%%
import numpy as np
import pandas as pd
from datetime import datetime
import holoviews as hv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from holoviews import opts
plt.style.use('seaborn-whitegrid')
hv.extension('plotly')

#%% Load data
import pandas as pd
shot_data = pd.read_csv('NBAShotSelection/data_kobe.csv')
shot_data.head()

#%% Add columns and shot (Game date to int)
shot_data['game_date_int'] = shot_data['game_date'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%d').toordinal())
# shot_data['game_time'] = (12 - shot_data['minutes_remaining']) + (shot_data['period'] - 1) * 12

shot_data.sort_values(['game_date_int', 'period', 'minutes_remaining', 'seconds_remaining'], ascending=[True, True, True, True])
shot_data.head()

#%% Split shots by year
shot_data['year'] = shot_data['game_date_int'].apply(
    lambda x: datetime.fromordinal(x).year)
u_years = np.unique(shot_data['year'])

#%% Latest 1000 shots animation

fig, ax = plt.subplots()

xdata, ydata = [], []
max_shots = 1000
block_size = 50
ln, = plt.plot([], [], marker='.', alpha=.5, ms=1, linestyle='None', animated=True)

def init():
    ax.set_xlim(-300, 300)
    ax.set_ylim(-50, 800)
    return ln,

def update(frame):
    inds = np.arange(frame * block_size, (frame + 1) * block_size)
    xdata.append(shot_data['loc_x'][inds])
    ydata.append(shot_data['loc_y'][inds])
    if len(xdata) > max_shots:
        del xdata[:block_size]
        del ydata[:block_size]
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.floor(shot_data.shape[0]/block_size),
                    init_func=init, blit=True)
ani.save('test.gif', fps=30)

#%% Yearly shots gif

fig, ax = plt.subplots()
fig.patch.set_facecolor('white')
ln, = plt.plot([], [], marker='.', alpha=.9, ms=2, linestyle='None', animated=True)

def init():
    ax.set_xlim(-300, 300)
    ax.set_ylim(-50, 800)
    return ln,

def update(year):
    filt = shot_data['year']==year
    xdata = shot_data['loc_x'].loc[filt]
    ydata = shot_data['loc_y'].loc[filt]
    ln.set_data(xdata, ydata)
    # ln.title(f'{year}')
    ax.set_title(f'{year}')
    print(f'{year}')
    return ln,

ani = FuncAnimation(fig, update, frames=u_years,
                    init_func=init, blit=True)
ani.save('yearly_shots.gif', fps=1, savefig_kwargs={'facecolor':'white'})












#%%
ds = hv.Dataset(shot_data, ['year']) #, ['Year', 'State'], vdims)

layout = (ds.to(hv.Scatter, 'loc_x', 'loc_y'))
layout
#.opts(
#    opts.Scatter(color='red', tools=['hover'])).cols(1)


#%%

#%%
plt.plot(shot_data['game_id'], shot_data['game_date_int'], '.')
plt.show()

#%%
np.sum(shot_data['game_date_int'] < 7.3e5)
np.sum(shot_data['game_date_int'] > 7.3e5)

#%%
plt.hist(shot_data['minutes_played'], bins=48)
plt.show()

#%%


#%%
shot_data['game_date_int']

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
