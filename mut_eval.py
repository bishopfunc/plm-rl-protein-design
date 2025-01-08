from constants import *
from policy import MutationPolicy, PositionPolicy
from plm_as_policy import PLMPolicy
from oracle_lib.metric import Evaluator
from oracle_lib.config import get_fitness_info
from mut_env import MutationEnv
from proxy import GFPScorer
from matplotlib import pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    wt_seq_aa = WT["GFP"]
    wt_seq_idx = seq_to_idx(wt_seq_aa)
      
    plm_policy = PLMPolicy()
    proxy = GFPScorer()
    mut_env = MutationEnv(wt_seq=wt_seq_aa, proxy=proxy)
    mut_policy = MutationPolicy(env=mut_env)
    mut_policy.load(path="mut_policy")
    
    sample_num = 1000
    mut_num = 10
    
    plm_policy_mut_aa_list = []
    mut_policy_mut_aa_list = []
    original_fitness_list = []
    plm_policy_fitness_list = []
    mut_policy_fitness_list = []
    plm_policy_improved_count = 0
    mut_policy_improved_count = 0
    
    length, min_fitness, max_fitness = get_fitness_info("GFP")
    evaluator = Evaluator(protein="GFP", max_target=max_fitness, min_target=min_fitness, device="cuda")
    
    for i in tqdm(range(sample_num)):
        pos = random.randint(0, len(wt_seq_aa) - 1)
        seq_aa = random_mutation(wt_seq_aa, mut_num)
        orog_fitness = proxy([seq_aa])[0]
        original_fitness_list.append(orog_fitness)
        seq_idx = seq_to_idx(seq_aa)
        # mut_policy
        mut, _ = mut_policy.predict({
            "sequence": seq_idx,
            "position": pos,
        })
        mut_aa = IDXTOAA[mut.item()]
        mut_seq_aa = seq_aa[:pos] + mut_aa + seq_aa[pos + 1:]
        mut_fitness = proxy([mut_seq_aa])[0]
        mut_policy_mut_aa_list.append(mut_aa)            
        mut_policy_fitness_list.append(mut_fitness)
        improved = mut_fitness > orog_fitness
        mut_policy_info = {"pos": pos, "mut": mut_aa, "fitness": mut_fitness, "improved": improved}
        mut_policy_improved_count += int(improved)
        
        # plm_policy
        mut_aa = plm_policy.get_mut(seq_aa, pos)
        mut_seq_aa = seq_aa[:pos] + mut_aa + seq_aa[pos + 1:]
        plm_fitness = proxy([mut_seq_aa])[0]            
        plm_policy_mut_aa_list.append(mut_aa)
        plm_policy_fitness_list.append(plm_fitness)
        improved = plm_fitness > orog_fitness
        plm_policy_info = {"pos": pos, "mut": mut_aa, "fitness": plm_fitness, "improved": improved}
        plm_policy_improved_count += int(improved)
        # print(f"{i=} {plm_policy_info=} {mut_policy_info=}")
      
            
    print(f"{plm_policy_improved_count=} {mut_policy_improved_count=}")
    x_list = [i for i in range(sample_num)]
    plt.plot(x_list, plm_policy_fitness_list, label="PLM Policy")
    plt.plot(x_list, mut_policy_fitness_list, label="Mutation Policy")
    plt.plot(x_list, original_fitness_list, label="Original")
    plt.legend()
    plt.savefig("fitness_comparison.png")
    plt.clf()
    
    plt.hist(plm_policy_mut_aa_list, bins=20, alpha=0.5, label="PLM Policy")
    plt.hist(mut_policy_mut_aa_list, bins=20, alpha=0.5, label="Mutation Policy")
    plt.legend()
    plt.savefig("mutation_comparison.png")
    plt.clf()
    
    
    
    
    
    
    