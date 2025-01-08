from mut_env import MutationEnv
from pos_env import PositionEnv
from policy import MutationPolicy, PositionPolicy
from constants import WT, idx_to_seq, seq_to_idx, IDXTOAA
from callback import CustomTQDMCallback, WandbLoggingCallback, SaveWeightsCallback
from proxy import GFPScorer
import warnings 
warnings.filterwarnings("ignore")
from wandb.integration.sb3 import WandbCallback
import os
import wandb
wandb.tensorboard.patch(root_logdir="tensorboard_logs")
run = wandb.init(project="PLMxRL", name="mut_train", monitor_gym=True)
      
if __name__ == "__main__":
    gfp_seq = WT["GFP"]
    env = MutationEnv(wt_seq=gfp_seq, proxy=GFPScorer())
    policy = MutationPolicy(env)
    total_timesteps = 1_000_000
    log_callback = WandbLoggingCallback(save_freq=1000)
    save_callback = SaveWeightsCallback(save_freq=1000, save_path='mut_policy')
    policy.learn(total_timesteps=total_timesteps, callback=[log_callback, save_callback])
    policy.save("mut_policy")
    
    # サンプリング
    obs, info = env.reset()
    print("初期観測:", idx_to_seq(obs["sequence"]))
    for _ in range(20):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, _, _, info = env.step(action)
        env.reset()
        pos = obs['position']
        mut_aa = IDXTOAA[action.item()]
        print(f"Action={mut_aa}, Reward={reward:.3f}, Obs={idx_to_seq(obs['sequence'])}")
    
wandb.finish()