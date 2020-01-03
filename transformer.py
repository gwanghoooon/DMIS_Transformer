import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

from layers import *

class EncoderLayer(nn.Module):
    
    def __init__(self, emb_dim, num_h, dropout = 0.1):

        super().__init__()

        self.norm_1 = Norm(emb_dim)
        self.norm_2 = Norm(emb_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(emb_dim,num_h)

        self.ff = FeedForward(emb_dim)

    def forward(self, x, mask):

        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))

        return x

class DecoderLayer(nn.Module):

    def __init__(self, emb_dim, num_h, dropout = 0.1):

        super().__init__()

        self.norm_1 = Norm(emb_dim)
        self.norm_2 = Norm(emb_dim)
        self.norm_3 = Norm(emb_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(emb_dim,num_h)
        self.attn_2 = MultiHeadAttention(emb_dim,num_h)

        self.ff = FeedForward(emb_dim).cuda()

    def forward(self, x, e_output, input_mask, output_mask):

        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2,x2,x2,output_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2,e_output,e_output,input_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))

        return x

class Encoder(nn.Module):

    def __init__(self, vocab_size, emb_dim, N, num_h):

        super().__init()

        self.N = N
        self.embed = InputEmbedding(vocab_size, emb_dim)
        self.layers = nn.ModuleList([copy.deepcopy(EncoderLayer(emb_dim,num_h)) for i in range(N)])
        self.norm = Norm(emb_dim)

    def forward(self, src, mask):
        
        x = self.embed(src)
        for i in range(self.N):
            x = self.layers[i](x,mask)

        return self.norm(x)


class Decoder(nn.Module):

    def __init__(self, vocab_size, emb_dim, N, num_h):

        super().__init__()

        self.N = N
        self.embed = InputEmbedding(vocab_size, emb_dim)
        self.layers = nn.ModuleList([copy.deepcopy(DecoderLayer(emb_dim,num_h)) for i in range(N)])
        self.norm = Norm(emb_dim)

    def forward(self, trg, e_outputs, src_mask, trg_mask):

        x = self.embed(trg)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)

        return self.norm(x)

class Transformer(nn.Module):

    def __init__(self, src_vocab, trg_vocab, emb_dim = 512, N = 6, num_h = 8):

        super().__init__()

        self.encoder = Encoder(src_vocab, emb_dim, N, num_h)
        self.decoder = Decoder(trg_vocab, emb_dim, N, num_h)
        self.out = nn.Linear(emb_dim, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):

        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)

        return output
