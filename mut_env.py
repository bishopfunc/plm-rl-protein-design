import gymnasium as gym
from gymnasium import spaces
import numpy as np
from constants import seq_to_idx, idx_to_seq, WT, random_mutation
from typing import Optional
from proxy import GFPScorer

class MutationEnv(gym.Env):
    """
    wt_seq: 配列の長さ（= 配列の各要素を アミノ酸番号 で表現）
    max_steps: 1エピソード内で実行できる最大ステップ数
    """
    def __init__(self, wt_seq: Optional[str] = None, proxy: Optional[GFPScorer] = None, max_steps: int = 20):
        super().__init__()
        self.wt_seq = wt_seq
        self.length = len(wt_seq)
        self.proxy = proxy
        self.proxy.setup() if proxy is not None else None # プロキシの初期化        
        self.max_mut_num = int(self.length * 0.30) # 変異数: 配列長の30%
        # 観測空間: アミノ酸配列 と 変異位置 の組み合わせ
        self.observation_space = spaces.Dict({
            "sequence": spaces.Box(low=0, high=19, shape=(self.length,), dtype=np.int32),
            "position": spaces.Discrete(self.length),
        })
        
        # 行動空間: 変異先のアミノ酸 = 0~19
        self.action_space = spaces.Discrete(20)

        # エピソードを初期化
        self.sequence = None
        self.pos = None
        self.max_steps = max_steps
        self.steps = 0
        
    def _calc_reward(self, sequence):
        """
        報酬の仮関数: 
        """
        if self.proxy is None:
            return 0.0
        return self.proxy(sequence)[0]

    def reset(self, seed=None, options=None):
        """
        環境をリセットして、初期状態(観測)を返す。
        """
        super().reset(seed=seed)

        mut_num = np.random.randint(1, self.max_mut_num)
        self.sequence = seq_to_idx(random_mutation(self.wt_seq, mut_num))

        self.pos = self.get_random_pos()
        self.steps = 0
        observation = {
            "sequence": self.sequence,
            "position": self.pos,
        }
        info = {}
        return observation, info
        
    def get_random_pos(self):
        return self.observation_space.sample()["position"]
    
    def step(self, action):
        """
        action = mut: 変異先のアミノ酸番号 (0 ~ 19)
        """        
        mut = action
         # 状態遷移におけるノイズのようなもので、ランダムに変異位置を選択
        self.sequence[self.pos] = mut  # 変異を適用
        self.steps += 1
  
        reward = self._calc_reward([idx_to_seq(self.sequence)])
        # done = True # 1ステップで終了
        done = (self.steps >= self.max_steps)
        truncated = False
      
        info = {}
        observation = {
            "sequence": self.sequence,
            "position": self.pos,
        }
        return observation, reward, done, truncated, info

if __name__ == "__main__":
    env = MutationEnv(wt_seq=WT["GFP"], proxy=GFPScorer())

    # 環境の初期化
    obs, info = env.reset()
    print("初期観測:", idx_to_seq(obs["sequence"]))

    while True:
        # ランダムな行動を取る
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Action={action}, Reward={reward:.3f}, Obs={idx_to_seq(obs['sequence'])}")
        if done:
            print("エピソード終了")
            break
