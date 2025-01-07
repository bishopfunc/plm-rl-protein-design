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

class WandbLoggingCallback(CustomTQDMCallback):
    def __init__(self, total_timesteps):
        super().__init__(total_timesteps)
        self.episode_rewards = []
        self.episode_losses = []

    def _on_step(self):
        # 毎ステップの報酬を収集
        reward = self.locals.get("rewards", None)
        if reward is not None:
            self.episode_rewards.append(reward)
        
        # 1エピソード終了時にログ
        if self.locals.get("dones", [False])[0]:
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            wandb.log({"mean_reward": mean_reward})
            self.episode_rewards = []  # リセット

        return super()._on_step()