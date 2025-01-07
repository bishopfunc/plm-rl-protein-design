from mut_env import MutationEnv
from pos_env import PositionEnv
from policy import MutationPolicy, PositionPolicy
from constants import WT, idx_to_seq, seq_to_idx
from callback import CustomTQDMCallback, WandbLoggingCallback
from proxy import GFPScorer
import warnings
warnings.filterwarnings("ignore")
from wandb.integration.sb3 import WandbCallback
import os
import wandb
run = wandb.init(project="PLMxRL", name="pos_train", monitor_gym=True)

class PositionPolicyEnvWrapper:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
    
    def train(self, total_timesteps=10000, callback=None):
        self.policy.learn(total_timesteps=total_timesteps, callback=callback)
      
      
if __name__ == "__main__":
    gfp_seq = WT["GFP"]
    env = PositionEnv(wt_seq=gfp_seq, proxy=GFPScorer())
    policy = PositionPolicy(env)
    wrapper = PositionPolicyEnvWrapper(env, policy)
    # total_timesteps = 1_000_000
    total_timesteps = 1
    tqdm_callback = WandbLoggingCallback(total_timesteps)
    os.makedirs("pos_policy", exist_ok=True)
    wandb_callback = WandbCallback(model_save_path='pos_policy')
    
    wrapper.train(total_timesteps=total_timesteps, callback=[tqdm_callback, wandb_callback])
    wrapper.policy.save("pos_policy")
    
    # サンプリング
    obs, info = env.reset()
    print("初期観測:", idx_to_seq(obs))
    for _ in range(20):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, _, _, info = env.step(action)
        env.reset()
        print(f"Action={action}, Reward={reward:.3f}, Obs={idx_to_seq(obs)}")
    
    
wandb.finish()