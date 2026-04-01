# Stable baselines


```python
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy


def measure_performance(model, env=None, n_episodes=20, deterministic=True):
    """
    Measures performance using an existing model and environment instance.
    :param model: The trained SB3 model instance (e.g., DQN, PPO, etc.)
    :param env: (Optional) A specific environment to evaluate on.
                If None, uses model.get_env().
    :param n_episodes: Number of episodes to evaluate.
    :param deterministic: Whether to use deterministic actions (recommended for eval).
    """
    # 1. Determine which environment to use
    eval_env = env if env is not None else model.get_env()
    if eval_env is None:
        print("Error: No environment provided and model has no environment attached.")
        return None
    print(f"--- Evaluating Performance ({n_episodes} episodes) ---")
    # 2. Run the evaluation
    # This returns a list of rewards and episode lengths
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        return_episode_rewards=True,
        deterministic=deterministic,
    )
    # 3. Calculate metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_len = np.mean(episode_lengths)
    # Example success threshold (adjust based on your specific task)
    success_rate = (np.array(episode_rewards) > 0).mean() * 100
    # 4. Print results
    print(f"Mean Reward:       {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Avg Episode Length: {mean_len:.1f} steps")
    print(f"Success Rate:      {success_rate:.1f}%")
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_len,
        "all_rewards": episode_rewards,
    }
```

## Cart pole


```python
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = DQN("MlpPolicy", vec_env, verbose=0, tensorboard_log="./dqn_tensorboard/")
```


```python
# 2. Training Loop
total_iterations = 1
steps_per_iteration = 100_000
for i in range(total_iterations):
    print(f"\n--- Iteration {i+1}/{total_iterations} ---")
    # CRITICAL: reset_num_timesteps=False
    # This ensures the internal step counter and learning rate
    # continue from where they left off instead of restarting at 0.
    model.learn(total_timesteps=steps_per_iteration, reset_num_timesteps=False)
    model.save("dqn_cartpole")

    # 3. Evaluate the current version
    # Note: evaluate_policy (inside measure_performance) resets the env
    stats = measure_performance(model, n_episodes=10)
    # Optional: Save checkpoint if performance is good
    # if stats['mean_reward'] > best_reward:
```

    
    --- Iteration 1/1 ---
    --- Evaluating Performance (10 episodes) ---
    Mean Reward:       500.00 +/- 0.00
    Avg Episode Length: 500.0 steps
    Success Rate:      100.0%



```python
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[29], line 5
          3 action, _states = model.predict(obs)
          4 obs, rewards, dones, info = vec_env.step(action)
    ----> 5 vec_env.render("human")


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py:104, in DummyVecEnv.render(self, mode)
         97 def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
         98     """
         99     Gym environment rendering. If there are multiple environments then
        100     they are tiled together in one image via ``BaseVecEnv.render()``.
        101 
        102     :param mode: The rendering type.
        103     """
    --> 104     return super().render(mode=mode)


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/base_vec_env.py:280, in VecEnv.render(self, mode)
        277     import cv2
        279     cv2.imshow("vecenv", bigimg[:, :, ::-1])
    --> 280     cv2.waitKey(1)
        281 else:
        282     return bigimg


    KeyboardInterrupt: 



    The Kernel crashed while executing code in the current cell or a previous cell. 


    Please review the code in the cell(s) to identify a possible cause of the failure. 


    Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. 


    View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.



```python
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = A2C("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model  # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
```

    Using cpu device
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 29.3     |
    |    ep_rew_mean        | 29.3     |
    | time/                 |          |
    |    fps                | 4338     |
    |    iterations         | 100      |
    |    time_elapsed       | 0        |
    |    total_timesteps    | 2000     |
    | train/                |          |
    |    entropy_loss       | -0.64    |
    |    explained_variance | 0.0574   |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 99       |
    |    policy_loss        | 1.64     |
    |    value_loss         | 7.78     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 41.4     |
    |    ep_rew_mean        | 41.4     |
    | time/                 |          |
    |    fps                | 5114     |
    |    iterations         | 200      |
    |    time_elapsed       | 0        |
    |    total_timesteps    | 4000     |
    | train/                |          |
    |    entropy_loss       | -0.587   |
    |    explained_variance | 0.0661   |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 199      |
    |    policy_loss        | 1.16     |
    |    value_loss         | 6.77     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 55.6     |
    |    ep_rew_mean        | 55.6     |
    | time/                 |          |
    |    fps                | 5868     |
    |    iterations         | 300      |
    |    time_elapsed       | 1        |
    |    total_timesteps    | 6000     |
    | train/                |          |
    |    entropy_loss       | -0.496   |
    |    explained_variance | 0.0269   |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 299      |
    |    policy_loss        | 1.29     |
    |    value_loss         | 6.05     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 71.8     |
    |    ep_rew_mean        | 71.8     |
    | time/                 |          |
    |    fps                | 6258     |
    |    iterations         | 400      |
    |    time_elapsed       | 1        |
    |    total_timesteps    | 8000     |
    | train/                |          |
    |    entropy_loss       | -0.507   |
    |    explained_variance | 0.00847  |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 399      |
    |    policy_loss        | 1.26     |
    |    value_loss         | 5.44     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 88       |
    |    ep_rew_mean        | 88       |
    | time/                 |          |
    |    fps                | 6507     |
    |    iterations         | 500      |
    |    time_elapsed       | 1        |
    |    total_timesteps    | 10000    |
    | train/                |          |
    |    entropy_loss       | -0.509   |
    |    explained_variance | 0.00258  |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 499      |
    |    policy_loss        | 0.999    |
    |    value_loss         | 4.81     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 107      |
    |    ep_rew_mean        | 107      |
    | time/                 |          |
    |    fps                | 6766     |
    |    iterations         | 600      |
    |    time_elapsed       | 1        |
    |    total_timesteps    | 12000    |
    | train/                |          |
    |    entropy_loss       | -0.461   |
    |    explained_variance | 0.00574  |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 599      |
    |    policy_loss        | 0.902    |
    |    value_loss         | 4.28     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 123      |
    |    ep_rew_mean        | 123      |
    | time/                 |          |
    |    fps                | 6957     |
    |    iterations         | 700      |
    |    time_elapsed       | 2        |
    |    total_timesteps    | 14000    |
    | train/                |          |
    |    entropy_loss       | -0.503   |
    |    explained_variance | 0.0223   |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 699      |
    |    policy_loss        | 0.852    |
    |    value_loss         | 3.78     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 136      |
    |    ep_rew_mean        | 136      |
    | time/                 |          |
    |    fps                | 6969     |
    |    iterations         | 800      |
    |    time_elapsed       | 2        |
    |    total_timesteps    | 16000    |
    | train/                |          |
    |    entropy_loss       | -0.462   |
    |    explained_variance | 0.00339  |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 799      |
    |    policy_loss        | 0.995    |
    |    value_loss         | 3.31     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 152      |
    |    ep_rew_mean        | 152      |
    | time/                 |          |
    |    fps                | 7116     |
    |    iterations         | 900      |
    |    time_elapsed       | 2        |
    |    total_timesteps    | 18000    |
    | train/                |          |
    |    entropy_loss       | -0.446   |
    |    explained_variance | -0.00198 |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 899      |
    |    policy_loss        | 0.363    |
    |    value_loss         | 2.82     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 169      |
    |    ep_rew_mean        | 169      |
    | time/                 |          |
    |    fps                | 7242     |
    |    iterations         | 1000     |
    |    time_elapsed       | 2        |
    |    total_timesteps    | 20000    |
    | train/                |          |
    |    entropy_loss       | -0.548   |
    |    explained_variance | 0.000551 |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 999      |
    |    policy_loss        | 0.618    |
    |    value_loss         | 2.39     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 179      |
    |    ep_rew_mean        | 179      |
    | time/                 |          |
    |    fps                | 7323     |
    |    iterations         | 1100     |
    |    time_elapsed       | 3        |
    |    total_timesteps    | 22000    |
    | train/                |          |
    |    entropy_loss       | -0.513   |
    |    explained_variance | 0.000646 |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 1099     |
    |    policy_loss        | 0.539    |
    |    value_loss         | 2        |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 194      |
    |    ep_rew_mean        | 194      |
    | time/                 |          |
    |    fps                | 7440     |
    |    iterations         | 1200     |
    |    time_elapsed       | 3        |
    |    total_timesteps    | 24000    |
    | train/                |          |
    |    entropy_loss       | -0.502   |
    |    explained_variance | 0.000789 |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 1199     |
    |    policy_loss        | 0.553    |
    |    value_loss         | 1.62     |
    ------------------------------------


    objc[45819]: Class SDLApplication is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0890) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead2c8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLAppDelegate is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f08e0) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead318). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLTranslatorResponder is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0958) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead390). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLMessageBoxPresenter is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0980) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead3b8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_cocoametalview is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f09d0) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead408). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLOpenGLContext is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0a20) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead458). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_ShapeData is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0a98) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead4d0). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_CocoaClosure is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0ae8) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead520). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_VideoData is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0b38) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead570). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_WindowData is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0b88) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead5c0). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLWindow is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0bb0) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead5e8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class Cocoa_WindowListener is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0bd8) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead610). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLView is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0c50) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead688). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class METAL_RenderData is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0cc8) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead700). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class METAL_TextureData is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0d18) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead750). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_RumbleMotor is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0d40) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead778). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_RumbleContext is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0d90) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib (0x165ead7c8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[1], line 19
         17 action, _states = model.predict(obs)
         18 obs, rewards, dones, info = vec_env.step(action)
    ---> 19 vec_env.render("human")


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py:104, in DummyVecEnv.render(self, mode)
         97 def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
         98     """
         99     Gym environment rendering. If there are multiple environments then
        100     they are tiled together in one image via ``BaseVecEnv.render()``.
        101 
        102     :param mode: The rendering type.
        103     """
    --> 104     return super().render(mode=mode)


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/base_vec_env.py:280, in VecEnv.render(self, mode)
        277     import cv2
        279     cv2.imshow("vecenv", bigimg[:, :, ::-1])
    --> 280     cv2.waitKey(1)
        281 else:
        282     return bigimg


    KeyboardInterrupt: 



```python
dir(model)
```




    ['__abstractmethods__',
     '__annotations__',
     '__class__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getstate__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__le__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__sizeof__',
     '__slots__',
     '__str__',
     '__subclasshook__',
     '__weakref__',
     '_abc_impl',
     '_current_progress_remaining',
     '_custom_logger',
     '_dump_logs',
     '_episode_num',
     '_excluded_save_params',
     '_get_policy_from_name',
     '_get_torch_save_params',
     '_init_callback',
     '_last_episode_starts',
     '_last_obs',
     '_last_original_obs',
     '_logger',
     '_maybe_recommend_cpu',
     '_n_updates',
     '_num_timesteps_at_start',
     '_setup_learn',
     '_setup_lr_schedule',
     '_setup_model',
     '_stats_window_size',
     '_total_timesteps',
     '_update_current_progress_remaining',
     '_update_info_buffer',
     '_update_learning_rate',
     '_vec_normalize_env',
     '_wrap_env',
     'action_noise',
     'action_space',
     'collect_rollouts',
     'device',
     'dump_logs',
     'ent_coef',
     'env',
     'ep_info_buffer',
     'ep_success_buffer',
     'gae_lambda',
     'gamma',
     'get_env',
     'get_parameters',
     'get_vec_normalize_env',
     'learn',
     'learning_rate',
     'load',
     'logger',
     'lr_schedule',
     'max_grad_norm',
     'n_envs',
     'n_steps',
     'normalize_advantage',
     'num_timesteps',
     'observation_space',
     'policy',
     'policy_aliases',
     'policy_class',
     'policy_kwargs',
     'predict',
     'rollout_buffer',
     'rollout_buffer_class',
     'rollout_buffer_kwargs',
     'save',
     'sde_sample_freq',
     'seed',
     'set_env',
     'set_logger',
     'set_parameters',
     'set_random_seed',
     'start_time',
     'tensorboard_log',
     'train',
     'use_sde',
     'verbose',
     'vf_coef']




```python
model.policy
```




    ActorCriticPolicy(
      (features_extractor): FlattenExtractor(
        (flatten): Flatten(start_dim=1, end_dim=-1)
      )
      (pi_features_extractor): FlattenExtractor(
        (flatten): Flatten(start_dim=1, end_dim=-1)
      )
      (vf_features_extractor): FlattenExtractor(
        (flatten): Flatten(start_dim=1, end_dim=-1)
      )
      (mlp_extractor): MlpExtractor(
        (policy_net): Sequential(
          (0): Linear(in_features=4, out_features=64, bias=True)
          (1): Tanh()
          (2): Linear(in_features=64, out_features=64, bias=True)
          (3): Tanh()
        )
        (value_net): Sequential(
          (0): Linear(in_features=4, out_features=64, bias=True)
          (1): Tanh()
          (2): Linear(in_features=64, out_features=64, bias=True)
          (3): Tanh()
        )
      )
      (action_net): Linear(in_features=64, out_features=2, bias=True)
      (value_net): Linear(in_features=64, out_features=1, bias=True)
    )




```python
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = A2C("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

# del model # remove to demonstrate saving and loading
```

    Using cpu device
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 22.3     |
    |    ep_rew_mean        | 22.3     |
    | time/                 |          |
    |    fps                | 6295     |
    |    iterations         | 100      |
    |    time_elapsed       | 0        |
    |    total_timesteps    | 2000     |
    | train/                |          |
    |    entropy_loss       | -0.592   |
    |    explained_variance | -0.00992 |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 99       |
    |    policy_loss        | 0.0171   |
    |    value_loss         | 23.1     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 24.5     |
    |    ep_rew_mean        | 24.5     |
    | time/                 |          |
    |    fps                | 7014     |
    |    iterations         | 200      |
    |    time_elapsed       | 0        |
    |    total_timesteps    | 4000     |
    | train/                |          |
    |    entropy_loss       | -0.568   |
    |    explained_variance | 0.687    |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 199      |
    |    policy_loss        | 1.45     |
    |    value_loss         | 6.76     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 38       |
    |    ep_rew_mean        | 38       |
    | time/                 |          |
    |    fps                | 7440     |
    |    iterations         | 300      |
    |    time_elapsed       | 0        |
    |    total_timesteps    | 6000     |
    | train/                |          |
    |    entropy_loss       | -0.571   |
    |    explained_variance | -0.0204  |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 299      |
    |    policy_loss        | 1.75     |
    |    value_loss         | 6.93     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 57.3     |
    |    ep_rew_mean        | 57.3     |
    | time/                 |          |
    |    fps                | 7742     |
    |    iterations         | 400      |
    |    time_elapsed       | 1        |
    |    total_timesteps    | 8000     |
    | train/                |          |
    |    entropy_loss       | -0.523   |
    |    explained_variance | -0.0402  |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 399      |
    |    policy_loss        | -1.32    |
    |    value_loss         | 68.1     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 73.7     |
    |    ep_rew_mean        | 73.7     |
    | time/                 |          |
    |    fps                | 7921     |
    |    iterations         | 500      |
    |    time_elapsed       | 1        |
    |    total_timesteps    | 10000    |
    | train/                |          |
    |    entropy_loss       | -0.575   |
    |    explained_variance | -0.0242  |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 499      |
    |    policy_loss        | 1.02     |
    |    value_loss         | 5.14     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 93.2     |
    |    ep_rew_mean        | 93.2     |
    | time/                 |          |
    |    fps                | 8063     |
    |    iterations         | 600      |
    |    time_elapsed       | 1        |
    |    total_timesteps    | 12000    |
    | train/                |          |
    |    entropy_loss       | -0.561   |
    |    explained_variance | 0.00256  |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 599      |
    |    policy_loss        | 1.19     |
    |    value_loss         | 4.47     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 110      |
    |    ep_rew_mean        | 110      |
    | time/                 |          |
    |    fps                | 8164     |
    |    iterations         | 700      |
    |    time_elapsed       | 1        |
    |    total_timesteps    | 14000    |
    | train/                |          |
    |    entropy_loss       | -0.53    |
    |    explained_variance | 0.00235  |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 699      |
    |    policy_loss        | 1.41     |
    |    value_loss         | 3.88     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 129      |
    |    ep_rew_mean        | 129      |
    | time/                 |          |
    |    fps                | 8250     |
    |    iterations         | 800      |
    |    time_elapsed       | 1        |
    |    total_timesteps    | 16000    |
    | train/                |          |
    |    entropy_loss       | -0.557   |
    |    explained_variance | 0.000713 |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 799      |
    |    policy_loss        | 0.862    |
    |    value_loss         | 3.31     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 149      |
    |    ep_rew_mean        | 149      |
    | time/                 |          |
    |    fps                | 8316     |
    |    iterations         | 900      |
    |    time_elapsed       | 2        |
    |    total_timesteps    | 18000    |
    | train/                |          |
    |    entropy_loss       | -0.59    |
    |    explained_variance | 0.000524 |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 899      |
    |    policy_loss        | 0.711    |
    |    value_loss         | 2.83     |
    ------------------------------------
    -------------------------------------
    | rollout/              |           |
    |    ep_len_mean        | 165       |
    |    ep_rew_mean        | 165       |
    | time/                 |           |
    |    fps                | 8293      |
    |    iterations         | 1000      |
    |    time_elapsed       | 2         |
    |    total_timesteps    | 20000     |
    | train/                |           |
    |    entropy_loss       | -0.539    |
    |    explained_variance | -0.000149 |
    |    learning_rate      | 0.0007    |
    |    n_updates          | 999       |
    |    policy_loss        | 0.645     |
    |    value_loss         | 2.38      |
    -------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 183      |
    |    ep_rew_mean        | 183      |
    | time/                 |          |
    |    fps                | 8343     |
    |    iterations         | 1100     |
    |    time_elapsed       | 2        |
    |    total_timesteps    | 22000    |
    | train/                |          |
    |    entropy_loss       | -0.462   |
    |    explained_variance | 0.000123 |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 1099     |
    |    policy_loss        | 0.598    |
    |    value_loss         | 1.98     |
    ------------------------------------
    ------------------------------------
    | rollout/              |          |
    |    ep_len_mean        | 202      |
    |    ep_rew_mean        | 202      |
    | time/                 |          |
    |    fps                | 8386     |
    |    iterations         | 1200     |
    |    time_elapsed       | 2        |
    |    total_timesteps    | 24000    |
    | train/                |          |
    |    entropy_loss       | -0.522   |
    |    explained_variance | 0.000111 |
    |    learning_rate      | 0.0007   |
    |    n_updates          | 1199     |
    |    policy_loss        | 0.665    |
    |    value_loss         | 1.63     |
    ------------------------------------



```python
model.action_space
```




    Discrete(2)




```python
# model = A2C.load("a2c_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[7], line 7
          5 action, _states = model.predict(obs)
          6 obs, rewards, dones, info = vec_env.step(action)
    ----> 7 vec_env.render("human")


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py:104, in DummyVecEnv.render(self, mode)
         97 def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
         98     """
         99     Gym environment rendering. If there are multiple environments then
        100     they are tiled together in one image via ``BaseVecEnv.render()``.
        101 
        102     :param mode: The rendering type.
        103     """
    --> 104     return super().render(mode=mode)


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/base_vec_env.py:271, in VecEnv.render(self, mode)
        267     return None
        269 if mode == "rgb_array" or mode == "human":
        270     # call the render method of the environments
    --> 271     images = self.get_images()
        272     # Create a big image by tiling images from subprocesses
        273     bigimg = tile_images(images)  # type: ignore[arg-type]


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py:95, in DummyVecEnv.get_images(self)
         91     warnings.warn(
         92         f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
         93     )
         94     return [None for _ in self.envs]
    ---> 95 return [env.render() for env in self.envs]


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/core.py:337, in Wrapper.render(self)
        335 def render(self) -> RenderFrame | list[RenderFrame] | None:
        336     """Uses the :meth:`render` of the :attr:`env` that can be overwritten to change the returned data."""
    --> 337     return self.env.render()


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/core.py:337, in Wrapper.render(self)
        335 def render(self) -> RenderFrame | list[RenderFrame] | None:
        336     """Uses the :meth:`render` of the :attr:`env` that can be overwritten to change the returned data."""
    --> 337     return self.env.render()


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/wrappers/common.py:409, in OrderEnforcing.render(self)
        404 if not self._disable_render_order_enforcing and not self._has_reset:
        405     raise ResetNeeded(
        406         "Cannot call `env.render()` before calling `env.reset()`, if this is an intended action, "
        407         "set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper."
        408     )
    --> 409 return super().render()


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/core.py:337, in Wrapper.render(self)
        335 def render(self) -> RenderFrame | list[RenderFrame] | None:
        336     """Uses the :meth:`render` of the :attr:`env` that can be overwritten to change the returned data."""
    --> 337     return self.env.render()


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/wrappers/common.py:303, in PassiveEnvChecker.render(self)
        301     return env_render_passive_checker(self.env)
        302 else:
    --> 303     return self.env.render()


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/envs/classic_control/cartpole.py:313, in CartPoleEnv.render(self)
        311     coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
        312     pole_coords.append(coord)
    --> 313 gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        314 gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))
        316 gfxdraw.aacircle(
        317     self.surf,
        318     int(cartx),
       (...)    321     (129, 132, 203),
        322 )


    KeyboardInterrupt: 


## Lunar lander


```python
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

env = gym.make("ALE/Breakout-v5", render_mode="human")


# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load("dqn_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">/Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packag
es/rich/live.py:260: UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
</pre>



    Using cpu device
    Wrapping the env with a `Monitor` wrapper
    Wrapping the env in a DummyVecEnv.
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 88       |
    |    ep_rew_mean      | -197     |
    |    exploration_rate | 0.983    |
    | time/               |          |
    |    episodes         | 4        |
    |    fps              | 1085     |
    |    time_elapsed     | 0        |
    |    total_timesteps  | 352      |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.886    |
    |    n_updates        | 62       |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 95.4     |
    |    ep_rew_mean      | -206     |
    |    exploration_rate | 0.964    |
    | time/               |          |
    |    episodes         | 8        |
    |    fps              | 1453     |
    |    time_elapsed     | 0        |
    |    total_timesteps  | 763      |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 7.23     |
    |    n_updates        | 165      |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 91.3     |
    |    ep_rew_mean      | -208     |
    |    exploration_rate | 0.948    |
    | time/               |          |
    |    episodes         | 12       |
    |    fps              | 1645     |
    |    time_elapsed     | 0        |
    |    total_timesteps  | 1096     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.629    |
    |    n_updates        | 248      |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 92.3     |
    |    ep_rew_mean      | -201     |
    |    exploration_rate | 0.93     |
    | time/               |          |
    |    episodes         | 16       |
    |    fps              | 1764     |
    |    time_elapsed     | 0        |
    |    total_timesteps  | 1477     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.814    |
    |    n_updates        | 344      |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 92.7     |
    |    ep_rew_mean      | -194     |
    |    exploration_rate | 0.912    |
    | time/               |          |
    |    episodes         | 20       |
    |    fps              | 1870     |
    |    time_elapsed     | 0        |
    |    total_timesteps  | 1853     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.71     |
    |    n_updates        | 438      |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 93       |
    |    ep_rew_mean      | -185     |
    |    exploration_rate | 0.894    |
    | time/               |          |
    |    episodes         | 24       |
    |    fps              | 1947     |
    |    time_elapsed     | 1        |
    |    total_timesteps  | 2232     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.05     |
    |    n_updates        | 532      |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 94.6     |
    |    ep_rew_mean      | -208     |
    |    exploration_rate | 0.874    |
    | time/               |          |
    |    episodes         | 28       |
    |    fps              | 2015     |
    |    time_elapsed     | 1        |
    |    total_timesteps  | 2648     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.27     |
    |    n_updates        | 636      |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 95.9     |
    |    ep_rew_mean      | -206     |
    |    exploration_rate | 0.854    |
    | time/               |          |
    |    episodes         | 32       |
    |    fps              | 2056     |
    |    time_elapsed     | 1        |
    |    total_timesteps  | 3068     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.33     |
    |    n_updates        | 741      |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 95.9     |
    |    ep_rew_mean      | -203     |
    |    exploration_rate | 0.836    |
    | time/               |          |
    |    episodes         | 36       |
    |    fps              | 2018     |
    |    time_elapsed     | 1        |
    |    total_timesteps  | 3451     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.432    |
    |    n_updates        | 837      |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 96       |
    |    ep_rew_mean      | -197     |
    |    exploration_rate | 0.818    |
    | time/               |          |
    |    episodes         | 40       |
    |    fps              | 1947     |
    |    time_elapsed     | 1        |
    |    total_timesteps  | 3840     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.13     |
    |    n_updates        | 934      |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 97       |
    |    ep_rew_mean      | -194     |
    |    exploration_rate | 0.797    |
    | time/               |          |
    |    episodes         | 44       |
    |    fps              | 1927     |
    |    time_elapsed     | 2        |
    |    total_timesteps  | 4268     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 10.1     |
    |    n_updates        | 1041     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 99.4     |
    |    ep_rew_mean      | -187     |
    |    exploration_rate | 0.773    |
    | time/               |          |
    |    episodes         | 48       |
    |    fps              | 1953     |
    |    time_elapsed     | 2        |
    |    total_timesteps  | 4769     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 6.52     |
    |    n_updates        | 1167     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 102      |
    |    ep_rew_mean      | -191     |
    |    exploration_rate | 0.748    |
    | time/               |          |
    |    episodes         | 52       |
    |    fps              | 1966     |
    |    time_elapsed     | 2        |
    |    total_timesteps  | 5298     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.779    |
    |    n_updates        | 1299     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 102      |
    |    ep_rew_mean      | -183     |
    |    exploration_rate | 0.729    |
    | time/               |          |
    |    episodes         | 56       |
    |    fps              | 1984     |
    |    time_elapsed     | 2        |
    |    total_timesteps  | 5708     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.54     |
    |    n_updates        | 1401     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 105      |
    |    ep_rew_mean      | -179     |
    |    exploration_rate | 0.701    |
    | time/               |          |
    |    episodes         | 60       |
    |    fps              | 1918     |
    |    time_elapsed     | 3        |
    |    total_timesteps  | 6296     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.13     |
    |    n_updates        | 1548     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 107      |
    |    ep_rew_mean      | -182     |
    |    exploration_rate | 0.674    |
    | time/               |          |
    |    episodes         | 64       |
    |    fps              | 1867     |
    |    time_elapsed     | 3        |
    |    total_timesteps  | 6854     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.92     |
    |    n_updates        | 1688     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 107      |
    |    ep_rew_mean      | -182     |
    |    exploration_rate | 0.655    |
    | time/               |          |
    |    episodes         | 68       |
    |    fps              | 1803     |
    |    time_elapsed     | 4        |
    |    total_timesteps  | 7273     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.7      |
    |    n_updates        | 1793     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 110      |
    |    ep_rew_mean      | -183     |
    |    exploration_rate | 0.626    |
    | time/               |          |
    |    episodes         | 72       |
    |    fps              | 1711     |
    |    time_elapsed     | 4        |
    |    total_timesteps  | 7884     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.504    |
    |    n_updates        | 1945     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 111      |
    |    ep_rew_mean      | -179     |
    |    exploration_rate | 0.598    |
    | time/               |          |
    |    episodes         | 76       |
    |    fps              | 1614     |
    |    time_elapsed     | 5        |
    |    total_timesteps  | 8466     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.553    |
    |    n_updates        | 2091     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 115      |
    |    ep_rew_mean      | -177     |
    |    exploration_rate | 0.563    |
    | time/               |          |
    |    episodes         | 80       |
    |    fps              | 1506     |
    |    time_elapsed     | 6        |
    |    total_timesteps  | 9207     |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 4.14     |
    |    n_updates        | 2276     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 119      |
    |    ep_rew_mean      | -173     |
    |    exploration_rate | 0.525    |
    | time/               |          |
    |    episodes         | 84       |
    |    fps              | 1465     |
    |    time_elapsed     | 6        |
    |    total_timesteps  | 10008    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.835    |
    |    n_updates        | 2476     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 124      |
    |    ep_rew_mean      | -172     |
    |    exploration_rate | 0.48     |
    | time/               |          |
    |    episodes         | 88       |
    |    fps              | 1415     |
    |    time_elapsed     | 7        |
    |    total_timesteps  | 10955    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 4.22     |
    |    n_updates        | 2713     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 126      |
    |    ep_rew_mean      | -170     |
    |    exploration_rate | 0.448    |
    | time/               |          |
    |    episodes         | 92       |
    |    fps              | 1388     |
    |    time_elapsed     | 8        |
    |    total_timesteps  | 11617    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.04     |
    |    n_updates        | 2879     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 140      |
    |    ep_rew_mean      | -169     |
    |    exploration_rate | 0.361    |
    | time/               |          |
    |    episodes         | 96       |
    |    fps              | 1154     |
    |    time_elapsed     | 11       |
    |    total_timesteps  | 13455    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.58     |
    |    n_updates        | 3338     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 144      |
    |    ep_rew_mean      | -169     |
    |    exploration_rate | 0.314    |
    | time/               |          |
    |    episodes         | 100      |
    |    fps              | 1137     |
    |    time_elapsed     | 12       |
    |    total_timesteps  | 14445    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.03     |
    |    n_updates        | 3586     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 161      |
    |    ep_rew_mean      | -166     |
    |    exploration_rate | 0.219    |
    | time/               |          |
    |    episodes         | 104      |
    |    fps              | 1047     |
    |    time_elapsed     | 15       |
    |    total_timesteps  | 16449    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 3.59     |
    |    n_updates        | 4087     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 180      |
    |    ep_rew_mean      | -163     |
    |    exploration_rate | 0.108    |
    | time/               |          |
    |    episodes         | 108      |
    |    fps              | 961      |
    |    time_elapsed     | 19       |
    |    total_timesteps  | 18779    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 4.47     |
    |    n_updates        | 4669     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 192      |
    |    ep_rew_mean      | -170     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 112      |
    |    fps              | 955      |
    |    time_elapsed     | 21       |
    |    total_timesteps  | 20306    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.36     |
    |    n_updates        | 5051     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 195      |
    |    ep_rew_mean      | -173     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 116      |
    |    fps              | 967      |
    |    time_elapsed     | 21       |
    |    total_timesteps  | 21003    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.67     |
    |    n_updates        | 5225     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 208      |
    |    ep_rew_mean      | -175     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 120      |
    |    fps              | 950      |
    |    time_elapsed     | 23       |
    |    total_timesteps  | 22674    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.36     |
    |    n_updates        | 5643     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 219      |
    |    ep_rew_mean      | -179     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 124      |
    |    fps              | 929      |
    |    time_elapsed     | 25       |
    |    total_timesteps  | 24084    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.727    |
    |    n_updates        | 5995     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 246      |
    |    ep_rew_mean      | -171     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 128      |
    |    fps              | 887      |
    |    time_elapsed     | 30       |
    |    total_timesteps  | 27277    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.602    |
    |    n_updates        | 6794     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 277      |
    |    ep_rew_mean      | -166     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 132      |
    |    fps              | 879      |
    |    time_elapsed     | 34       |
    |    total_timesteps  | 30726    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.24     |
    |    n_updates        | 7656     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 290      |
    |    ep_rew_mean      | -167     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 136      |
    |    fps              | 890      |
    |    time_elapsed     | 36       |
    |    total_timesteps  | 32413    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.729    |
    |    n_updates        | 8078     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 304      |
    |    ep_rew_mean      | -174     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 140      |
    |    fps              | 895      |
    |    time_elapsed     | 38       |
    |    total_timesteps  | 34197    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.551    |
    |    n_updates        | 8524     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 317      |
    |    ep_rew_mean      | -175     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 144      |
    |    fps              | 903      |
    |    time_elapsed     | 39       |
    |    total_timesteps  | 36009    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.823    |
    |    n_updates        | 8977     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 346      |
    |    ep_rew_mean      | -176     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 148      |
    |    fps              | 868      |
    |    time_elapsed     | 45       |
    |    total_timesteps  | 39367    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.825    |
    |    n_updates        | 9816     |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 377      |
    |    ep_rew_mean      | -167     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 152      |
    |    fps              | 863      |
    |    time_elapsed     | 49       |
    |    total_timesteps  | 42973    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.779    |
    |    n_updates        | 10718    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 413      |
    |    ep_rew_mean      | -168     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 156      |
    |    fps              | 840      |
    |    time_elapsed     | 55       |
    |    total_timesteps  | 46973    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.743    |
    |    n_updates        | 11718    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 441      |
    |    ep_rew_mean      | -165     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 160      |
    |    fps              | 842      |
    |    time_elapsed     | 59       |
    |    total_timesteps  | 50411    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.739    |
    |    n_updates        | 12577    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 473      |
    |    ep_rew_mean      | -158     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 164      |
    |    fps              | 839      |
    |    time_elapsed     | 64       |
    |    total_timesteps  | 54110    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.646    |
    |    n_updates        | 13502    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 507      |
    |    ep_rew_mean      | -149     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 168      |
    |    fps              | 840      |
    |    time_elapsed     | 68       |
    |    total_timesteps  | 57954    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.737    |
    |    n_updates        | 14463    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 528      |
    |    ep_rew_mean      | -138     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 172      |
    |    fps              | 846      |
    |    time_elapsed     | 71       |
    |    total_timesteps  | 60667    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.16     |
    |    n_updates        | 15141    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 535      |
    |    ep_rew_mean      | -137     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 176      |
    |    fps              | 856      |
    |    time_elapsed     | 72       |
    |    total_timesteps  | 61920    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.738    |
    |    n_updates        | 15454    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 544      |
    |    ep_rew_mean      | -137     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 180      |
    |    fps              | 865      |
    |    time_elapsed     | 73       |
    |    total_timesteps  | 63639    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.53     |
    |    n_updates        | 15884    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 575      |
    |    ep_rew_mean      | -135     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 184      |
    |    fps              | 857      |
    |    time_elapsed     | 78       |
    |    total_timesteps  | 67459    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.483    |
    |    n_updates        | 16839    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 597      |
    |    ep_rew_mean      | -137     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 188      |
    |    fps              | 857      |
    |    time_elapsed     | 82       |
    |    total_timesteps  | 70625    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.79     |
    |    n_updates        | 17631    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 621      |
    |    ep_rew_mean      | -128     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 192      |
    |    fps              | 862      |
    |    time_elapsed     | 85       |
    |    total_timesteps  | 73700    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.539    |
    |    n_updates        | 18399    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 635      |
    |    ep_rew_mean      | -124     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 196      |
    |    fps              | 860      |
    |    time_elapsed     | 89       |
    |    total_timesteps  | 76935    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.547    |
    |    n_updates        | 19208    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 659      |
    |    ep_rew_mean      | -116     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 200      |
    |    fps              | 853      |
    |    time_elapsed     | 94       |
    |    total_timesteps  | 80336    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.853    |
    |    n_updates        | 20058    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 664      |
    |    ep_rew_mean      | -113     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 204      |
    |    fps              | 860      |
    |    time_elapsed     | 96       |
    |    total_timesteps  | 82895    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.07     |
    |    n_updates        | 20698    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 675      |
    |    ep_rew_mean      | -105     |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 208      |
    |    fps              | 861      |
    |    time_elapsed     | 100      |
    |    total_timesteps  | 86320    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 4.49     |
    |    n_updates        | 21554    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 686      |
    |    ep_rew_mean      | -85.8    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 212      |
    |    fps              | 866      |
    |    time_elapsed     | 102      |
    |    total_timesteps  | 88895    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.511    |
    |    n_updates        | 22198    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 702      |
    |    ep_rew_mean      | -74.6    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 216      |
    |    fps              | 869      |
    |    time_elapsed     | 104      |
    |    total_timesteps  | 91246    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.817    |
    |    n_updates        | 22786    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 707      |
    |    ep_rew_mean      | -66.3    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 220      |
    |    fps              | 873      |
    |    time_elapsed     | 106      |
    |    total_timesteps  | 93377    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.726    |
    |    n_updates        | 23319    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 714      |
    |    ep_rew_mean      | -54      |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 224      |
    |    fps              | 879      |
    |    time_elapsed     | 108      |
    |    total_timesteps  | 95464    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.676    |
    |    n_updates        | 23840    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 710      |
    |    ep_rew_mean      | -51.6    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 228      |
    |    fps              | 878      |
    |    time_elapsed     | 111      |
    |    total_timesteps  | 98307    |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.935    |
    |    n_updates        | 24551    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 698      |
    |    ep_rew_mean      | -52.8    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 232      |
    |    fps              | 882      |
    |    time_elapsed     | 113      |
    |    total_timesteps  | 100486   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.65     |
    |    n_updates        | 25096    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 716      |
    |    ep_rew_mean      | -48.4    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 236      |
    |    fps              | 883      |
    |    time_elapsed     | 117      |
    |    total_timesteps  | 103982   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.63     |
    |    n_updates        | 25970    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 731      |
    |    ep_rew_mean      | -40.4    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 240      |
    |    fps              | 878      |
    |    time_elapsed     | 122      |
    |    total_timesteps  | 107325   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.64     |
    |    n_updates        | 26806    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 738      |
    |    ep_rew_mean      | -33.7    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 244      |
    |    fps              | 881      |
    |    time_elapsed     | 124      |
    |    total_timesteps  | 109791   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.744    |
    |    n_updates        | 27422    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 730      |
    |    ep_rew_mean      | -26.8    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 248      |
    |    fps              | 881      |
    |    time_elapsed     | 127      |
    |    total_timesteps  | 112408   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.65     |
    |    n_updates        | 28076    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 727      |
    |    ep_rew_mean      | -23.4    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 252      |
    |    fps              | 880      |
    |    time_elapsed     | 131      |
    |    total_timesteps  | 115701   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.1      |
    |    n_updates        | 28900    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 720      |
    |    ep_rew_mean      | -21.8    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 256      |
    |    fps              | 877      |
    |    time_elapsed     | 135      |
    |    total_timesteps  | 118964   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.571    |
    |    n_updates        | 29715    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 712      |
    |    ep_rew_mean      | -26.1    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 260      |
    |    fps              | 876      |
    |    time_elapsed     | 138      |
    |    total_timesteps  | 121611   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.744    |
    |    n_updates        | 30377    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 711      |
    |    ep_rew_mean      | -24.8    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 264      |
    |    fps              | 879      |
    |    time_elapsed     | 142      |
    |    total_timesteps  | 125233   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.795    |
    |    n_updates        | 31283    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 713      |
    |    ep_rew_mean      | -29      |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 268      |
    |    fps              | 873      |
    |    time_elapsed     | 147      |
    |    total_timesteps  | 129233   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.784    |
    |    n_updates        | 32283    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 726      |
    |    ep_rew_mean      | -36.5    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 272      |
    |    fps              | 863      |
    |    time_elapsed     | 154      |
    |    total_timesteps  | 133233   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.61     |
    |    n_updates        | 33283    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 753      |
    |    ep_rew_mean      | -37.4    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 276      |
    |    fps              | 862      |
    |    time_elapsed     | 159      |
    |    total_timesteps  | 137233   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.977    |
    |    n_updates        | 34283    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 776      |
    |    ep_rew_mean      | -37.2    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 280      |
    |    fps              | 858      |
    |    time_elapsed     | 164      |
    |    total_timesteps  | 141233   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.555    |
    |    n_updates        | 35283    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 778      |
    |    ep_rew_mean      | -40      |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 284      |
    |    fps              | 852      |
    |    time_elapsed     | 170      |
    |    total_timesteps  | 145233   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.674    |
    |    n_updates        | 36283    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 786      |
    |    ep_rew_mean      | -37.9    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 288      |
    |    fps              | 845      |
    |    time_elapsed     | 176      |
    |    total_timesteps  | 149233   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.696    |
    |    n_updates        | 37283    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 793      |
    |    ep_rew_mean      | -42.7    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 292      |
    |    fps              | 842      |
    |    time_elapsed     | 181      |
    |    total_timesteps  | 152980   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 4.58     |
    |    n_updates        | 38219    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 800      |
    |    ep_rew_mean      | -46.2    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 296      |
    |    fps              | 838      |
    |    time_elapsed     | 187      |
    |    total_timesteps  | 156980   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.664    |
    |    n_updates        | 39219    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 806      |
    |    ep_rew_mean      | -51.4    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 300      |
    |    fps              | 835      |
    |    time_elapsed     | 192      |
    |    total_timesteps  | 160980   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.5      |
    |    n_updates        | 40219    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 821      |
    |    ep_rew_mean      | -53.5    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 304      |
    |    fps              | 832      |
    |    time_elapsed     | 198      |
    |    total_timesteps  | 164980   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.467    |
    |    n_updates        | 41219    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 827      |
    |    ep_rew_mean      | -60.2    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 308      |
    |    fps              | 835      |
    |    time_elapsed     | 202      |
    |    total_timesteps  | 168980   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.04     |
    |    n_updates        | 42219    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 837      |
    |    ep_rew_mean      | -64      |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 312      |
    |    fps              | 837      |
    |    time_elapsed     | 205      |
    |    total_timesteps  | 172605   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.759    |
    |    n_updates        | 43126    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 854      |
    |    ep_rew_mean      | -70.5    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 316      |
    |    fps              | 837      |
    |    time_elapsed     | 210      |
    |    total_timesteps  | 176605   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.788    |
    |    n_updates        | 44126    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 872      |
    |    ep_rew_mean      | -74.5    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 320      |
    |    fps              | 836      |
    |    time_elapsed     | 215      |
    |    total_timesteps  | 180605   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.725    |
    |    n_updates        | 45126    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 891      |
    |    ep_rew_mean      | -81.6    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 324      |
    |    fps              | 834      |
    |    time_elapsed     | 221      |
    |    total_timesteps  | 184605   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.409    |
    |    n_updates        | 46126    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 903      |
    |    ep_rew_mean      | -83      |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 328      |
    |    fps              | 832      |
    |    time_elapsed     | 226      |
    |    total_timesteps  | 188605   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.596    |
    |    n_updates        | 47126    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 920      |
    |    ep_rew_mean      | -84.6    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 332      |
    |    fps              | 834      |
    |    time_elapsed     | 230      |
    |    total_timesteps  | 192437   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 1.88     |
    |    n_updates        | 48084    |
    ----------------------------------
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 925      |
    |    ep_rew_mean      | -85.5    |
    |    exploration_rate | 0.05     |
    | time/               |          |
    |    episodes         | 336      |
    |    fps              | 829      |
    |    time_elapsed     | 236      |
    |    total_timesteps  | 196437   |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 3.83     |
    |    n_updates        | 49084    |
    ----------------------------------



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Wrapping the env with a `Monitor` wrapper
    Wrapping the env in a DummyVecEnv.


## Breakout


```python
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os

# --- CONFIGURATION ---
ENV_ID = "ALE/Breakout-v5"
SAVE_DIR = "./logs/dqn_checkpoints/"
TOTAL_STEPS = 200000
SAVE_FREQ = 50000  # Save a version every 50,000 steps
# Create directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)
# 1. Setup Environment
# Note: For Atari games like Breakout, CnnPolicy is usually better than MlpPolicy
env = gym.make(ENV_ID, render_mode="human")
# --- PHASE 1: TRAINING WITH CHECKPOINTS ---
# This callback will save files like: dqn_model_50000_steps.zip, dqn_model_100000_steps.zip
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ, save_path=SAVE_DIR, name_prefix="dqn_model"
)
print(f"Starting training for {TOTAL_STEPS} steps...")
model = DQN("CnnPolicy", env, verbose=1)  # Changed to CnnPolicy as Breakout uses pixels
model.learn(
    total_timesteps=TOTAL_STEPS, callback=checkpoint_callback, progress_bar=True
)
model.save(os.path.join(SAVE_DIR, "dqn_model_final"))
print("Training complete. Checkpoints saved in:", SAVE_DIR)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Starting training for 200000 steps...
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Using cpu device
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Wrapping the env with a `Monitor` wrapper
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Wrapping the env in a DummyVecEnv.
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Wrapping the env in a VecTransposeImage.
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">/Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packag
es/stable_baselines3/common/buffers.py:242: UserWarning: This system does not have apparently enough memory to 
store the complete replay buffer 201.62GB &gt; 1.35GB
  warnings.warn(
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">----------------------------------
| rollout/            |          |
|    ep_len_mean      | 175      |
|    ep_rew_mean      | 1        |
|    exploration_rate | 0.967    |
| time/               |          |
|    episodes         | 4        |
|    fps              | 12       |
|    time_elapsed     | 58       |
|    total_timesteps  | 701      |
| train/              |          |
|    learning_rate    | 0.0001   |
|    loss             | 2.54e-05 |
|    n_updates        | 150      |
----------------------------------
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">----------------------------------
| rollout/            |          |
|    ep_len_mean      | 173      |
|    ep_rew_mean      | 1        |
|    exploration_rate | 0.934    |
| time/               |          |
|    episodes         | 8        |
|    fps              | 11       |
|    time_elapsed     | 121      |
|    total_timesteps  | 1386     |
| train/              |          |
|    learning_rate    | 0.0001   |
|    loss             | 7.52e-05 |
|    n_updates        | 321      |
----------------------------------
</pre>




    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[29], line 26
         24 print(f"Starting training for {TOTAL_STEPS} steps...")
         25 model = DQN("CnnPolicy", env, verbose=1) # Changed to CnnPolicy as Breakout uses pixels
    ---> 26 model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_callback, progress_bar=True)
         27 model.save(os.path.join(SAVE_DIR, "dqn_model_final"))
         28 print("Training complete. Checkpoints saved in:", SAVE_DIR)


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/dqn/dqn.py:272, in DQN.learn(self, total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)
        263 def learn(
        264     self: SelfDQN,
        265     total_timesteps: int,
       (...)    270     progress_bar: bool = False,
        271 ) -> SelfDQN:
    --> 272     return super().learn(
        273         total_timesteps=total_timesteps,
        274         callback=callback,
        275         log_interval=log_interval,
        276         tb_log_name=tb_log_name,
        277         reset_num_timesteps=reset_num_timesteps,
        278         progress_bar=progress_bar,
        279     )


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/off_policy_algorithm.py:335, in OffPolicyAlgorithm.learn(self, total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)
        332 assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()
        334 while self.num_timesteps < total_timesteps:
    --> 335     rollout = self.collect_rollouts(
        336         self.env,
        337         train_freq=self.train_freq,
        338         action_noise=self.action_noise,
        339         callback=callback,
        340         learning_starts=self.learning_starts,
        341         replay_buffer=self.replay_buffer,
        342         log_interval=log_interval,
        343     )
        345     if not rollout.continue_training:
        346         break


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/off_policy_algorithm.py:568, in OffPolicyAlgorithm.collect_rollouts(self, env, callback, train_freq, replay_buffer, action_noise, learning_starts, log_interval)
        565 actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
        567 # Rescale and perform action
    --> 568 new_obs, rewards, dones, infos = env.step(actions)
        570 self.num_timesteps += env.num_envs
        571 num_collected_steps += 1


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/base_vec_env.py:222, in VecEnv.step(self, actions)
        215 """
        216 Step the environments with the given action
        217 
        218 :param actions: the action
        219 :return: observation, reward, done, information
        220 """
        221 self.step_async(actions)
    --> 222 return self.step_wait()


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/vec_transpose.py:97, in VecTransposeImage.step_wait(self)
         96 def step_wait(self) -> VecEnvStepReturn:
    ---> 97     observations, rewards, dones, infos = self.venv.step_wait()
         99     # Transpose the terminal observations
        100     for idx, done in enumerate(dones):


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py:59, in DummyVecEnv.step_wait(self)
         56 def step_wait(self) -> VecEnvStepReturn:
         57     # Avoid circular imports
         58     for env_idx in range(self.num_envs):
    ---> 59         obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(  # type: ignore[assignment]
         60             self.actions[env_idx]
         61         )
         62         # convert to SB3 VecEnv api
         63         self.buf_dones[env_idx] = terminated or truncated


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/monitor.py:94, in Monitor.step(self, action)
         92 if self.needs_reset:
         93     raise RuntimeError("Tried to step environment that needs reset")
    ---> 94 observation, reward, terminated, truncated, info = self.env.step(action)
         95 self.rewards.append(float(reward))
         96 if terminated or truncated:


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/wrappers/common.py:393, in OrderEnforcing.step(self, action)
        391 if not self._has_reset:
        392     raise ResetNeeded("Cannot call env.step() before calling env.reset()")
    --> 393 return super().step(action)


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/core.py:327, in Wrapper.step(self, action)
        323 def step(
        324     self, action: WrapperActType
        325 ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        326     """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
    --> 327     return self.env.step(action)


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/wrappers/common.py:285, in PassiveEnvChecker.step(self, action)
        283     return env_step_passive_checker(self.env, action)
        284 else:
    --> 285     return self.env.step(action)


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/env.py:305, in AtariEnv.step(self, action)
        303 reward = 0.0
        304 for _ in range(frameskip):
    --> 305     reward += self.ale.act(action_idx, strength)
        307 is_terminal = self.ale.game_over(with_truncation=False)
        308 is_truncated = self.ale.game_truncated()


    KeyboardInterrupt: 



```python

```




```python
model.replay_buffer.
```




    array([[[[[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]]]],
    
    
    
           [[[[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]]]],
    
    
    
           [[[[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]]]],
    
    
    
           ...,
    
    
    
           [[[[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]]]],
    
    
    
           [[[[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]]]],
    
    
    
           [[[[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]],
    
             [[0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              ...,
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0],
              [0, 0, 0, ..., 0, 0, 0]]]]],
          shape=(1000000, 1, 3, 210, 160), dtype=uint8)




```python
# --- PHASE 2: LOADING A SPECIFIC VERSION ---
# Change this string to the specific step count you want to test
VERSION_TO_LOAD = "dqn_model_100000_steps.zip"
load_path = os.path.join(SAVE_DIR, VERSION_TO_LOAD)
if os.path.exists(load_path):
    print(f"Loading version: {VERSION_TO_LOAD}")
    # Load the specific checkpoint
    model = DQN.load(load_path, env=env)
    # Evaluate the specific version
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    print(f"Mean reward for {VERSION_TO_LOAD}: {mean_reward:.2f} +/- {std_reward:.2f}")
    # --- SIMULATE ---
    print("Starting simulation...")
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
else:
    print(f"Error: Could not find checkpoint at {load_path}")
```


```python

```


```python

```


```python
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import ale_py

gym.register_envs(ale_py)

# Create environment
env = gym.make("ALE/Breakout-v5", render_mode="human")


# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)
# Save the agent
model.save("dqn_breakout")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load("dqn_breakout", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
```

    A.L.E: Arcade Learning Environment (version 0.11.2+ecc1138)
    [Powered by Stella]
    objc[45819]: Class SDL_RumbleMotor is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0d40) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90910). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">/Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packag
es/rich/live.py:260: UserWarning: install "ipywidgets" for Jupyter support
  warnings.warn('install "ipywidgets" for Jupyter support')
</pre>



    objc[45819]: Class SDL_RumbleContext is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0d90) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90960). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLApplication is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0890) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de909b0). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLAppDelegate is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f08e0) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90a00). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLTranslatorResponder is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0958) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90a78). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLMessageBoxPresenter is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0980) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90aa0). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_cocoametalview is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f09d0) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90af0). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLOpenGLContext is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0a20) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90b40). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_ShapeData is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0a98) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90bb8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_CocoaClosure is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0ae8) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90c08). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_VideoData is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0b38) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90c58). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDL_WindowData is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0b88) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90ca8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLWindow is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0bb0) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90cd0). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class Cocoa_WindowListener is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0bd8) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90cf8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class SDLView is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0c50) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90d70). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class METAL_RenderData is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0cc8) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90de8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    objc[45819]: Class METAL_TextureData is implemented in both /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib (0x1615f0d18) and /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/libSDL2-2.0.0.dylib (0x17de90e38). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
    /Users/rich/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/buffers.py:242: UserWarning: This system does not have apparently enough memory to store the complete replay buffer 201.62GB > 1.18GB
      warnings.warn(


    Using cpu device
    Wrapping the env with a `Monitor` wrapper
    Wrapping the env in a DummyVecEnv.
    Wrapping the env in a VecTransposeImage.
    ----------------------------------
    | rollout/            |          |
    |    ep_len_mean      | 174      |
    |    ep_rew_mean      | 1        |
    |    exploration_rate | 0.967    |
    | time/               |          |
    |    episodes         | 4        |
    |    fps              | 14       |
    |    time_elapsed     | 47       |
    |    total_timesteps  | 697      |
    | train/              |          |
    |    learning_rate    | 0.0001   |
    |    loss             | 0.000649 |
    |    n_updates        | 149      |
    ----------------------------------



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[28], line 17
         15 model = DQN("MlpPolicy", env, verbose=1)
         16 # Train the agent and display a progress bar
    ---> 17 model.learn(total_timesteps=int(2e5), progress_bar=True)
         18 # Save the agent
         19 model.save("dqn_breakout")


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/dqn/dqn.py:272, in DQN.learn(self, total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)
        263 def learn(
        264     self: SelfDQN,
        265     total_timesteps: int,
       (...)    270     progress_bar: bool = False,
        271 ) -> SelfDQN:
    --> 272     return super().learn(
        273         total_timesteps=total_timesteps,
        274         callback=callback,
        275         log_interval=log_interval,
        276         tb_log_name=tb_log_name,
        277         reset_num_timesteps=reset_num_timesteps,
        278         progress_bar=progress_bar,
        279     )


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/off_policy_algorithm.py:335, in OffPolicyAlgorithm.learn(self, total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)
        332 assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()
        334 while self.num_timesteps < total_timesteps:
    --> 335     rollout = self.collect_rollouts(
        336         self.env,
        337         train_freq=self.train_freq,
        338         action_noise=self.action_noise,
        339         callback=callback,
        340         learning_starts=self.learning_starts,
        341         replay_buffer=self.replay_buffer,
        342         log_interval=log_interval,
        343     )
        345     if not rollout.continue_training:
        346         break


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/off_policy_algorithm.py:568, in OffPolicyAlgorithm.collect_rollouts(self, env, callback, train_freq, replay_buffer, action_noise, learning_starts, log_interval)
        565 actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
        567 # Rescale and perform action
    --> 568 new_obs, rewards, dones, infos = env.step(actions)
        570 self.num_timesteps += env.num_envs
        571 num_collected_steps += 1


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/base_vec_env.py:222, in VecEnv.step(self, actions)
        215 """
        216 Step the environments with the given action
        217 
        218 :param actions: the action
        219 :return: observation, reward, done, information
        220 """
        221 self.step_async(actions)
    --> 222 return self.step_wait()


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/vec_transpose.py:97, in VecTransposeImage.step_wait(self)
         96 def step_wait(self) -> VecEnvStepReturn:
    ---> 97     observations, rewards, dones, infos = self.venv.step_wait()
         99     # Transpose the terminal observations
        100     for idx, done in enumerate(dones):


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py:59, in DummyVecEnv.step_wait(self)
         56 def step_wait(self) -> VecEnvStepReturn:
         57     # Avoid circular imports
         58     for env_idx in range(self.num_envs):
    ---> 59         obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(  # type: ignore[assignment]
         60             self.actions[env_idx]
         61         )
         62         # convert to SB3 VecEnv api
         63         self.buf_dones[env_idx] = terminated or truncated


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/stable_baselines3/common/monitor.py:94, in Monitor.step(self, action)
         92 if self.needs_reset:
         93     raise RuntimeError("Tried to step environment that needs reset")
    ---> 94 observation, reward, terminated, truncated, info = self.env.step(action)
         95 self.rewards.append(float(reward))
         96 if terminated or truncated:


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/wrappers/common.py:393, in OrderEnforcing.step(self, action)
        391 if not self._has_reset:
        392     raise ResetNeeded("Cannot call env.step() before calling env.reset()")
    --> 393 return super().step(action)


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/core.py:327, in Wrapper.step(self, action)
        323 def step(
        324     self, action: WrapperActType
        325 ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        326     """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
    --> 327     return self.env.step(action)


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/gymnasium/wrappers/common.py:285, in PassiveEnvChecker.step(self, action)
        283     return env_step_passive_checker(self.env, action)
        284 else:
    --> 285     return self.env.step(action)


    File ~/Developer/Github/VariousDataAnalysis/reinforcement_learning/q_learning/.venv/lib/python3.12/site-packages/ale_py/env.py:305, in AtariEnv.step(self, action)
        303 reward = 0.0
        304 for _ in range(frameskip):
    --> 305     reward += self.ale.act(action_idx, strength)
        307 is_terminal = self.ale.game_over(with_truncation=False)
        308 is_truncated = self.ale.game_truncated()


    KeyboardInterrupt: 



```python
rewards
```




    array([-0.12929237], dtype=float32)




```python
model.get_env()
```




    <stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv at 0x17d9823f0>




```python
env
```




    <TimeLimit<OrderEnforcing<PassiveEnvChecker<LunarLander<LunarLander-v3>>>>>




```python
# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
```


```python
obs
```




    array([[ 0.00149927,  1.4190471 ,  0.15185723,  0.36119175, -0.00173061,
            -0.03439793,  0.        ,  0.        ]], dtype=float32)




```python
vec_env.
```




    array([0])


