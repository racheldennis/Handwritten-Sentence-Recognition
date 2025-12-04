import torch

class Vocab:
    def __init__(self, vocab, blank="-"):
        self.blank = blank
        
        self.char2idx = {blank: 0}
        self.char2idx.update({c: i+1 for i, c in enumerate(vocab)})

        self.idx2char = {i: c for c, i in self.char2idx.items()}
    
    def __len__(self):
        return len(self.char2idx)
    
    def encode(self, label):
        return [self.char2idx[c] for c in label if c in self.char2idx]
    
    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().tolist()

        decoded = []
        prev_index = None

        for i in seq:
            i = int(i)
            if i != prev_index and i != 0:
                decoded.append(self.idx2char[i])
            prev_index = i
        
        return "".join(decoded)
