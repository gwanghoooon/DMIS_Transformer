import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

def position_encoding_init(emb_dim, n_position = 82 ):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim

    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def attention(q, k, v, d_k, mask = None):

    #dropout?

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill_(mask, -1e9) # true일때 -1e9로 설정

    scores = torch.matmul(q, k.transpose(-2,-1))
    scores = F.softmax(scores, dim=-1)

    return torch.matmul(scores, v)

class InputEmbedding(nn.Module):

    def __init__(self, vocab_size, emb_dim):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pe = position_encoding_init(emb_dim)

    def forward(self, x):

        x - self.embed(x)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad = False).cuda()

        return x

class MultiHeadAttention(nn.module):

    def __init__(self, emb_dim, num_h):

        super().__init__()

        self.num_h = num_h
        self.emb_dim = emb_dim
        self.d_k = emb_dim / num_h

        self.linear_q = nn.Linear(emb_dim,emb_dim)
        self.linear_k = nn.Linear(emb_dim,emb_dim)
        self.linear_v = nn.Linear(emb_dim,emb_dim)

        self.linear_o = nn.Linear(emb_dim,emb_dim)


    def forward(self, q, k, v, mask):

        batchSize = q.size(0)

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v) 

        q = q.view(batchSize, -1, self.num_h, self.d_k).transpose(1,2) # split head
        k = k.view(batchSize, -1, self.num_h, self.d_k).transpose(1,2) # split head
        v = v.view(batchSize, -1, self.num_h, self.d_k).transpose(1,2) # split head

        o = attention(q, k, v, self.d_k, mask) # self attention

        o = o.transpose(1,2).contiguous().view(batchSize, -1, self.emb_dim) # concat

        o = self.linear_o(o) 

        return o


class FeedForward(nn.module):

    def __init__(self,emb_dim,ff_dim = 2048, dropout = 0.1):

        super().__init__()

        self.linear_1 = nn.Linear(emb_dim,ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim,emb_dim)

    def forward(self,x,e_output,mask):

        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


class Norm(nn.Module):

    def __init__(self, emb_dim, eps = 1e-6):

        super().__init__()
    
        self.size = emb_dim
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):

        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

        return norm