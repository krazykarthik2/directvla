import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from clip.simple_tokenizer import SimpleTokenizer

_bpe_tokenizer = SimpleTokenizer()

def tokenize_prompt(text: str, context_length: int = 77):
    return clip.tokenize([text], context_length=context_length, truncate=True)[0]

def decode_tokens(token_ids):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    words = []
    for tid in token_ids:
        if tid == 0:
            continue
        try:
            decoded = _bpe_tokenizer.decode([tid]).strip()
            if decoded:
                words.append(decoded)
        except Exception:
            words.append(f'<tok_{tid}>')
    return words

class SmolVLMTokenizer:
    def __init__(self):
        self.tokenizer = _bpe_tokenizer
        self.context_length = 77
        self.pad_id = 0
        self.bos_id = 49406
        self.eos_id = 49407

    def encode(self, text: str, max_len: int = 77):
        return clip.tokenize([text], context_length=max_len, truncate=True)[0]

    def decode_active_tokens(self, token_tensor):
        t_list = token_tensor.tolist() if isinstance(token_tensor, torch.Tensor) else list(token_tensor)
        words = []
        for tid in t_list:
            if tid == 0:
                continue
            txt = _bpe_tokenizer.decode([tid]).strip()
            if txt:
                words.append(txt)
        return words

if __name__ == '__main__':
    tok = SmolVLMTokenizer()
    prompt = 'pick up the red cube and place it on the green platform'
    t_ids = tok.encode(prompt)
    print('Tokenized prompt tensor shape:', t_ids.shape)
    words = tok.decode_active_tokens(t_ids)
    print('Decoded words:', words)
