import pandas as pd
from baseline.insilico import Model
import argparse
from config import get_fitness_info
import torch
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--protein', type=str, choices=['GFP', 'AAV'], required=True)
parser.add_argument('--level', type=str, choices=['hard', 'medium'], required=True)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], required=True)
args = parser.parse_args()


protein = args.protein 
level = args.level

length, min_fitness, max_fitness = get_fitness_info(protein) 

def main():
  starting_sequences = pd.read_csv(f'data/{args.protein}/{args.level}.csv')
  starting_sequences.rename(columns={'target': 'true_score'}, inplace=True)
  starting_sequences = starting_sequences[['sequence', 'true_score']]
  starting_sequences['true_score'] = (starting_sequences['true_score'] - min_fitness)/(max_fitness - min_fitness)

  oracle = Model(epochs=10, device=args.device)
  oracle_ckpt = torch.load(f'ckpt/{protein}/oracle.ckpt', map_location=args.device)
  if "state_dict" in oracle_ckpt.keys():
      oracle_ckpt = oracle_ckpt["state_dict"]
  oracle.model.load_state_dict({ k.replace('predictor.',''):v for k,v in oracle_ckpt.items() })
  
  starting_sequences['oracle_score'] = oracle.get_fitness(starting_sequences['sequence'].values)
  starting_sequences['oracle_score'] = (starting_sequences['oracle_score'] - min_fitness)/(max_fitness - min_fitness)
  starting_sequences.to_csv("test.csv", index=False)
  print(f"saved to ./test.csv")
  # print(starting_sequences[['true_score', 'oracle_score']])
if __name__ == '__main__':
  main()