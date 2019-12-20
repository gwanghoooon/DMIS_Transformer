import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers import InputEmbedding, MultiHeadAttention, FeedForward, Norm

class EncoderLayer(nn.Module):
    
    def __init__(self, emb_dim, num_h, dropout = 0.1):

        super().__init__()

        self.norm_1 = Norm(emb_dim)
        self.norm_2 = Norm(emb_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(emb_dim,num_h)

        self.ff = FeedForward(emb_dim)

    def forward(self, x):

        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2))
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
