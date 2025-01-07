import gymnasium as gym
from gymnasium import spaces
import numpy as np
from constants import seq_to_idx, idx_to_seq, WT, ALPHABET, AATOIDX
from typing import Optional
from proxy import GFPScorer
from plm_as_policy import PLMPolicy
from policy import MutationPolicy, PositionPolicy
from mut_env import MutationEnv

def mock_mutation_policy():
    """
    モックの方策関数: ランダムな変異を返す
    """
    return np.random.randint(20)

class PositionEnv(gym.Env):
    """
    wt_seq: 配列の長さ（= 配列の各要素を アミノ酸番号 で表現）
    max_steps: 1エピソード内で実行できる最大ステップ数
    """
    def __init__(self, wt_seq: Optional[str] = None, proxy: Optional[GFPScorer] = None, max_steps: int = 20):
        super().__init__()
        self.wt_seq = wt_seq
        self.length = len(wt_seq)
        self.max_steps = max_steps
        
        # 観測空間: アミノ酸配列 各要素が 0~19 のいずれか
        self.observation_space = spaces.Box(low=0, high=19, shape=(self.length,), dtype=np.int32)
        
        # 行動空間: 変異位置 = 0~length-1
        self.action_space = spaces.Discrete(self.length)

        # エピソードを初期化
        self.sequence = None
        self.steps = 0
        self.model = PLMPolicy()
        # mut_env = MutationEnv(wt_seq=wt_seq, proxy=proxy)
        # self.model = MutationPolicy.load(path="mut_policy", policy="MultiInputPolicy", env=mut_env)
        self.proxy = proxy
        self.proxy.setup() if proxy is not None else None # プロキシの初期化        
        
    def _calc_reward(self, state):
        """
        報酬の仮関数: 
        """
        if self.proxy is None:
            return 0.0
        return self.proxy(state)[0]

    def reset(self, seed=None, options=None):
        """
        環境をリセットして、初期状態(観測)を返す。
        """
        super().reset(seed=seed)

        self.sequence = seq_to_idx(self.wt_seq)
        self.steps = 0

        # 観測として、現在の配列を返す
        observation = self.sequence
        info = {}
        return observation, info
        
    def get_random_pos(self):
        return np.random.randint(self.length)
    
    def step(self, action):
        """
        action = pos: 変異位置 (0 ~ length-1)
        """
        pos = action
        obs = {
            "sequence": self.sequence,
            "position": 0,
        }        
        # print(f"Current sequence: {obs}")
        # mut = self.model.predict(obs)
        # print(f"Predicted mutation: {mut}")
        mut = AATOIDX[self.model.get_mut([idx_to_seq(self.sequence)], pos)]
        self.sequence[pos] = mut
        self.steps += 1
        
        reward = self._calc_reward([idx_to_seq(self.sequence)])
        done = (self.steps >= self.max_steps)
        truncated = False

        observation = self.sequence
        info = {}
        return observation, reward, done, truncated, info

if __name__ == "__main__":
    env = PositionEnv(wt_seq=WT["GFP"], proxy=GFPScorer(), max_steps=20)

    # 環境の初期化
    obs, info = env.reset()
    print("初期観測:", idx_to_seq(obs))

    while True:
        # ランダムな行動を取る
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action={action}, Reward={reward:.3f}, Obs={idx_to_seq(obs)}")

        if done:
            print("エピソード終了")
            break
