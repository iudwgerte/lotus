import random

import numpy as np
import torch
from transformers import AutoTokenizer

from model.lotus import Lotus
from model.lotus_config import LotusConfig

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_model():
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/lotus_tokenizer")
    model = Lotus(LotusConfig())
    model = model.to("cuda")

    print(f'Model parameters: {count_parameters(model) / 1e6} million = {count_parameters(model) / 1e9} B (Billion)')
    return model, tokenizer

if __name__ == '__main__':
    setup_seed(42)
    model, tokenizer = init_model()
    """
    text = tokenizer.bos_token + "I saw a shaddow on the wall."
    x = tokenizer(text).data['input_ids']
    x = torch.tensor(x, dtype=torch.long, device="cuda")[None, :]
    with torch.no_grad():
        res_y = model.generate(x, tokenizer.eos_token_id, 512, temperature=5, stream=True)
        try:
            y = next(res_y)
        except StopIteration:
            print("No output generated")
            exit()
        
        hist_idx = 0

        while y != None:
            answer = tokenizer.decode(y[0].tolist())
            if answer and answer[-1] == '�':
                try:
                    y = next(res_y)
                except:
                    break
                continue
            
            if not len(answer):
                try:
                    y = next(res_y)
                except:
                    break
                continue

            print(answer[hist_idx:], end='', flush=True)
            try:
                y = next(res_y)
            except:
                break
            hist_idx = len(answer)"""