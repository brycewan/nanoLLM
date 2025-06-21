import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0 # Ensure d_model is divisible by n_heads
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.attn = None
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1) Linear projections: [batch, seq_len, d_model] -> [batch, n_head, seq_len, d_k]
        query = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2) Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) # Dot product
        scores = scores / (self.d_k ** 0.5) # Scaling
        if mask is not None: # Apply mask if provided
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        self.attn = attn  # Save attention weights
        output = torch.matmul(attn, value) # Weighted sum
        
        # 3) Concatenate heads and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.W_O(output)
    
    
class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_hiden, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_hiden)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(ffn_hiden, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))
    
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias
    
    
class SubLayerConnection(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # Pre-LN & residual connection
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hiden, dropout=0.1):
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.sublayer1 = SubLayerConnection(d_model, dropout)
        self.ffn = FeedForward(d_model, ffn_hiden, dropout)
        self.sublayer2 = SubLayerConnection(d_model, dropout)
        
    def forward(self, x, mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer2(x, self.ffn)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layer, n_blocks):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_blocks)])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hiden, dropout=0.1):
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.src_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.sublayer1 = SubLayerConnection(d_model, dropout)
        self.sublayer2 = SubLayerConnection(d_model, dropout)
        self.ffn = FeedForward(d_model, ffn_hiden, dropout)
        self.sublayer3 = SubLayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer2(x, lambda x: self.src_attn(x, memory, memory, src_mask))
        x = self.sublayer3(x, self.ffn)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layer, n_blocks):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_blocks)])
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
    
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(maxlen, d_model)
        position = torch.arange(0, maxlen).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen=5000, dropout=0.1):
        super(Embedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, dropout, maxlen)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        return self.dropout(self.token_embedding(x) + self.positional_embedding(x))