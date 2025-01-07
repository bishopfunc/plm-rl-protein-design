from policy import MutationPolicy, PositionPolicy
from constants import WT, idx_to_seq, seq_to_idx
from callback import CustomTQDMCallback, WandbLoggingCallback
from mut_env import MutationEnv
from pos_env import PositionEnv
from proxy import GFPScorer
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    gfp_seq = WT["GFP"]
    proxy = GFPScorer()
    pos_env = PositionEnv(wt_seq=gfp_seq, proxy=proxy)
    pos_policy = PositionPolicy.load(path="pos_policy",  env=pos_env)
    mut_env = MutationEnv(wt_seq=gfp_seq, proxy=proxy)
    mut_policy = MutationPolicy.load(path="mut_policy", env=mut_env)
    
    max_steps = 20
    gfp_seq = seq_to_idx(WT["GFP"])
    for _ in range(max_steps):
        pos_probs, _ = pos_policy.predict(gfp_seq, deterministic=True)
        pos = pos_probs.argmax()
        mut_probs, _ = mut_policy.predict({
            "sequence": gfp_seq,
            "position": pos,
        }, deterministic=True)
        mut = mut_probs.argmax()
        gfp_seq[pos] = mut
        print(f"Pos={pos}, Mut={mut} Reward={proxy([idx_to_seq(gfp_seq)])[0]:.3f}, Obs={idx_to_seq(gfp_seq)}")
    
    