from stable_baselines3 import PPO
import torch as th
import os

class PositionPolicy:
    def __init__(self, env):
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs={
                "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
                "activation_fn": th.nn.ReLU,
            },
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
        )
        
    def load(self, path):
        self.model = PPO.load(path)
    
    def save(self, path):
        self.model.save(path)
        
    def learn(self, total_timesteps=10000, callback=None):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
    def predict(self, input, deterministic=True):
        return self.model.predict(input, deterministic=deterministic)

class MutationPolicy:
    def __init__(self, env):
        self.model = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs={
                "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
                "activation_fn": th.nn.ReLU,
            },
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
        )
        
    def load(self, path):
        self.model = PPO.load(path)
    
    def save(self, path):
        self.model.save(path)
        
    def learn(self, total_timesteps=10000, callback=None):
        self.model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        
    def predict(self, input, deterministic=True):
        return self.model.predict(input, deterministic=deterministic)

# class PositionPolicy(PPO):
#     def __init__(self, env, *args, **kwargs):
#         super(PositionPolicy, self).__init__(
#             *args, **kwargs,
#             policy="MlpPolicy",
#             env=env,
#             # policy_kwargs={
#             #     "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
#             #     "activation_fn": th.nn.ReLU,
#             # },
#             verbose=1,
#         )

# class MutationPolicy(PPO):
#     def __init__(self, env, *args, **kwargs):
#         super(MutationPolicy, self).__init__(
#             *args, **kwargs,            
#             policy="MultiInputPolicy",
#             env=env,
#             # policy_kwargs={
#             #     "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
#             #     "activation_fn": th.nn.ReLU,
#             # },
#             verbose=1,
#         )
        
