from stable_baselines3 import PPO
import torch as th

class PositionPolicy(PPO):
    def __init__(self, env):
        super(PositionPolicy, self).__init__(
            policy="MlpPolicy",
            env=env,
            policy_kwargs={
                "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
                "activation_fn": th.nn.ReLU,
            },
            verbose=1,
        )

class MutationPolicy(PPO):
    def __init__(self, env):
        super(MutationPolicy, self).__init__(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs={
                "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
                "activation_fn": th.nn.ReLU,
            },
            verbose=1,
        )
        
