from policy import MutationPolicy, PositionPolicy
from constants import WT, idx_to_seq, seq_to_idx, seq_to_one_hot, IDXTOAA
from callback import CustomTQDMCallback, WandbLoggingCallback
from mut_env import MutationEnv
from pos_env import PositionEnv
from proxy import GFPScorer
from oracle_lib.metric import Evaluator
import warnings
warnings.filterwarnings("ignore")
import random
import torch
from oracle_lib.config import get_fitness_info
import pandas as pd

if __name__ == "__main__":
    gfp_seq = WT["GFP"]
    proxy = GFPScorer()
    pos_env = PositionEnv(wt_seq=gfp_seq, proxy=proxy)
    pos_policy = PositionPolicy(env=pos_env)
    pos_policy.load(path="pos_policy")
    mut_env = MutationEnv(wt_seq=gfp_seq, proxy=proxy)
    mut_policy = MutationPolicy(env=mut_env)
    mut_policy.load(path="mut_policy/model_weights_50000")
    
    length, min_fitness, max_fitness = get_fitness_info("GFP")
    evaluator = Evaluator(protein="GFP", max_target=max_fitness, min_target=min_fitness, device="cuda")
    
    sequences = []
    sequences_aa = []
    sequences_onehot = []
    
    max_steps = 10_000_000
    log_interval = 100
    gfp_seq_aa = WT["GFP"]
    gfp_seq_onehot = seq_to_one_hot(gfp_seq_aa)
    gfp_seq = seq_to_idx(gfp_seq_aa)
    
    data = pd.read_csv(f'/home/ubuntu/workspace/plm-rl-protein-design/oracle_lib/data/GFP/hard.csv')[["sequence", "target"]]
    inits = data["sequence"].tolist()
    for step in range(max_steps):
        with torch.no_grad():
            pos, _ = pos_policy.predict(gfp_seq, deterministic=True)
            mut, _ = mut_policy.predict({
                "sequence": gfp_seq,
                "position": pos,
            }, deterministic=True)
        # pos = pos.item()
        pos = random.randint(0, len(gfp_seq)-1)
        mut = mut.item()
        gfp_seq[pos] = mut
        sequences_aa.append(idx_to_seq(gfp_seq))
        sequences_onehot.append(seq_to_one_hot(sequences_aa[-1]))
        if step % log_interval == 0:
            gfp_seq_aa = idx_to_seq(gfp_seq)
            print(f"Step={step} Pos={pos}, Mut={IDXTOAA[mut]} Reward={proxy([idx_to_seq(gfp_seq)])[0]:.3f}")
            # print(f"Step={step} Pos={pos}, Mut={IDXTOAA[mut]} Reward={proxy([idx_to_seq(gfp_seq)])[0]:.3f}, Obs={gfp_seq_aa}")
            fitness, diversity, novelty, high = evaluator.evaluate(sequences_aa, inits)
            print(f"Fitness={fitness:.3f}, Diversity={diversity:.3f}, Novelty={novelty:.3f}, High={high:.3f}")
           
            
    
    