from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
from constants import ALPHABET

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

class SaveWeightsCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super(SaveWeightsCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 現在のエピソード数を確認
        if self.n_calls % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f'model_weights_{self.n_calls}.zip')
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"Model weights saved to {save_file}")
        return True
        
class WandbLoggingCallback(BaseCallback):
    def __init__(self, save_freq, verbose=0):
        super().__init__(verbose=verbose)
        self.episode_rewards = []
        self.total_positions = []
        self.total_actions = []
        self.save_freq = save_freq
        self.heat_map = None
        self.sequences = []
        self.class_labels = list(range(20))
        
    def _on_step(self):
        # 毎ステップの報酬を収集
        reward = self.locals.get("rewards", None)
        if reward is not None:
            self.episode_rewards.append(reward)

        # 毎ステップのログを記録
        obs = self.locals.get("new_obs", None)
        if obs is not None:
            self.total_positions.append(obs["position"].item())
            self.sequence = obs["sequence"][0]
            self.sequences.append(self.sequence)

        actions = self.locals.get("actions", None)
        if actions is not None:
            self.total_actions.append(actions.item())
        
        # 1エピソード終了時にログ
        if self.locals.get("dones", [False])[0]:
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
            if self.n_calls % self.save_freq == 0:
                # ヒートマップを作成                
                data = np.array(self.sequences)
                self.heat_map = np.array([[np.sum(data[col, :] == label) for col in range(data.shape[-1])] for label in self.class_labels])
                plt.figure(figsize=(15, 5))
                plt.yticks(range(20), ALPHABET)
                plt.imshow(self.heat_map, cmap="viridis", aspect="auto")
                plt.colorbar(label='Intensity')  # カラーバーを追加                    
                plt.savefig("heatmap.png")
                plt.clf()
                    
            # wandbにログを記録
            wandb.log({
                "mean_reward": mean_reward,
                "total_positions": wandb.Histogram(self.total_positions),
                "total_actions": wandb.Histogram(self.total_actions),
                "heatmap": wandb.Image("heatmap.png"),
            })
            # print(f"mean_reward: {mean_reward}")
            # print(f"total_positions: {self.total_positions}")
            # print(f"total_actions: {self.total_actions}")
            # リセット
            self.episode_rewards = []

        return True
