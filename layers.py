import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Inputembedding(nn.Module):

    def __init__(self, vocab_size, emb_dim = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x):
        return self.embed(x)
        # Positional embedding 추가!


