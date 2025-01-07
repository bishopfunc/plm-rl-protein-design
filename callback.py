from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import wandb

class CustomTQDMCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        """Training start: Initialize tqdm progress bar."""
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="step")

    def _on_step(self) -> bool:
        """Step: Update tqdm progress bar."""
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        """Training end: Close tqdm progress bar."""
        self.pbar.close()

class WandbLoggingCallback(BaseCallback):
    def __init__(self, total_timesteps=None, verbose=0):
        super().__init__(verbose=verbose)
        self.total_timesteps = total_timesteps

        self.episode_rewards = []
        self.total_positions = []
        self.total_actions = []
        
    def _on_step(self):
        # 毎ステップの報酬を収集
        reward = self.locals.get("rewards", None)
        if reward is not None:
            self.episode_rewards.append(reward)

        # 毎ステップのログを記録
        obs = self.locals.get("new_obs", None)
        if obs is not None:
            self.total_positions.append(obs["position"].item())
        
        actions = self.locals.get("actions", None)
        if actions is not None:
            self.total_actions.append(actions.item())
        
        # 1エピソード終了時にログ
        if self.locals.get("dones", [False])[0]:
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
            # wandbにログを記録
            wandb.log({
                "mean_reward": mean_reward,
                "total_positions": wandb.Histogram(self.total_positions),
                "total_actions": wandb.Histogram(self.total_actions)
            })
            # print(f"mean_reward: {mean_reward}")
            # print(f"total_positions: {self.total_positions}")
            # print(f"total_actions: {self.total_actions}")
            # リセット
            self.episode_rewards = []

        return True
