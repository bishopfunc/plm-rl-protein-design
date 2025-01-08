import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from constants import WT, ALPHABET
import random

class PLMScorer:
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name)
        # ESMのアミノ酸アルファベットを取得

    def get_llr(self, sequence: str, pos: int, mut_aa: str) -> float:
        # 入力配列をトークナイズ
        encoding = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            padding=True, 
            add_special_tokens=True
        )
        wt_ids= self.tokenizer.convert_tokens_to_ids(sequence[pos])
        mut_ids = self.tokenizer.convert_tokens_to_ids(mut_aa)
        
        # ESMトークナイザーは特殊トークン（<cls>, <eos>）を含むため位置を補正
        pos = pos + 1  
        
        # 入力トークンIDを取得し、対象位置をマスク
        input_ids = encoding.input_ids  # shape: [1, seq_len+2]
        
        masked_input_ids = input_ids.clone()
        masked_input_ids[:, pos] = self.tokenizer.mask_token_id  # 対象位置を<mask>に置き換え
        
        with torch.no_grad():
            outputs = self.model(masked_input_ids)
            logits = outputs.logits  # shape: [1, seq_len+2, vocab_size]
        # 対象位置のトークンのスコアを取得
        prob_wt = logits[:, pos, wt_ids]
        prob_mut = logits[:, pos, mut_ids]
        llr = torch.log(prob_mut/prob_wt + 1e-8)
        if torch.isnan(llr):
            return 0.0
        return llr.item()

if __name__ == "__main__":
    scorer = PLMScorer()
    wt_seq = WT["GFP"] 
    pos = random.randint(0, len(wt_seq)-1)
    mut_aa = random.choice(ALPHABET)
    wt_aa = wt_seq[pos]
    llr = scorer.get_llr(wt_seq, pos, mut_aa)
    print(f"pos: {pos}, {wt_aa}->{mut_aa}, LLR: {llr:.3f}")
    