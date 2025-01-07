from typing import List, Union
from oracle_lib.config import get_fitness_info
from oracle_lib.baseline.insilico import Model 
from torchtyping import TensorType
import numpy as np
import torch
from constants import WT, idx_to_seq, seq_to_idx


class GFPScorer:
    """
    input: amino acid sequence (string)
    output: gfp cnn score (float)
    """
    def __init__(self):
        self.oracle = None
        self.protein = "GFP"
        self.device = "cuda"
        self.min_fitness = None
        self.max_fitness = None
        self.length = None
    
    def setup(self):
        self.oracle = Model(epochs=10, device=self.device)
        oracle_ckpt = torch.load(f'oracle_lib/ckpt/{self.protein}/oracle.ckpt', map_location=self.device)
        if "state_dict" in oracle_ckpt.keys():
            oracle_ckpt = oracle_ckpt["state_dict"]
        self.oracle.model.load_state_dict({ k.replace('predictor.',''):v for k,v in oracle_ckpt.items() }) 
        self.length, self.min_fitness, self.max_fitness = get_fitness_info(self.protein)
    
    def normalize_fitness(self, fitness: List[float]) -> torch.Tensor:
        return (fitness - self.min_fitness)/(self.max_fitness - self.min_fitness)
        
    def __call__(
        self, states: List[str]) -> List[float]:
        fitnesss = self.oracle.get_fitness(states)  
        fitnesss = self.normalize_fitness(fitnesss)
        return fitnesss.tolist()

if __name__ == "__main__":
    scorer = GFPScorer()
    scorer.setup()
    gfp_seq = WT["GFP"]
    score = scorer([gfp_seq] * 3)
    print(score)