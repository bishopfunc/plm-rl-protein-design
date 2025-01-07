import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from constants import WT
import random

class PLMPolicy:
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name)
        # ESMのアミノ酸アルファベットを取得

    def get_mut(self, sequence: str, pos: int) -> str:
        # 入力配列をトークナイズ
        encoding = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            padding=True, 
            add_special_tokens=True
        )
        
        # ESMトークナイザーは特殊トークン（<cls>, <eos>）を含むため位置を補正
        pos = pos + 1  
        
        # 入力トークンIDを取得し、対象位置をマスク
        input_ids = encoding.input_ids  # shape: [1, seq_len+2]
        masked_input_ids = input_ids.clone()
        masked_input_ids[:, pos] = self.tokenizer.mask_token_id  # 対象位置を<mask>に置き換え
        
        with torch.no_grad():
            outputs = self.model(masked_input_ids)
            logits = outputs.logits  # shape: [1, seq_len+2, vocab_size]
        
        # ロジットをソフトマックスで変換
        probs = torch.softmax(logits[0, pos], dim=0)
        
        # トークンIDをスコア順にソート
        sorted_indices = torch.argsort(probs, descending=True)
        
        # スコアが高い順にアミノ酸に対応するトークンを探す
        for token_id in sorted_indices:
            token_id = token_id.item()
            mut_token = self.tokenizer.convert_ids_to_tokens(token_id)
            if mut_token in self.tokenizer.get_vocab():
                return mut_token
        
        return "X"  # 該当するアミノ酸が見つからない場合

if __name__ == "__main__":
    policy = PLMPolicy()
    wt_seq = WT["GFP"] 
    pos = random.randint(0, len(wt_seq)-1)
    mut = policy.get_mut(wt_seq, pos)
    print(f"wildtype: {wt_seq}, position: {pos}, mutation: {mut}")
