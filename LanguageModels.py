import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, unique_character_count):
        super().__init__()

        self.token_embedding_table = nn.Embedding(unique_character_count, unique_character_count)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:

            B, T, C = logits.shape

            # reshape to fit PyTorch's cross_entropy function (from 3d to 2d & 2d to 1d
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            #evaluate log fuction using pytorch cross entropy function
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

def make_model(xb, yb, unique_character_count):
    m = BigramLanguageModel(unique_character_count)
    logits, loss = m(xb, yb)
    print(logits.shape)

    print("Loss:", loss)
    print("Target Loss", -(math.log(1/logits.shape[1])))

    print(main.decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].toList()))