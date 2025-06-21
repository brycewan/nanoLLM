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
        # self.attn = None
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # self.dropout = nn.Dropout(p=dropout)
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
        # attn = self.dropout(attn)
        # self.attn = attn  # Save attention weights
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
        super(EncoderLayer, self).__init__()
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
        super(DecoderLayer, self).__init__()
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
        self.d_model = d_model
        
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
        pe = pe.unsqueeze(0) # Add batch dimension, [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)].requires_grad_(False)
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen=5000, dropout=0.1):
        super(Embedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, maxlen)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        return self.dropout(self.token_embedding(x) + self.positional_embedding(x))
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
    
class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 n_head=2, 
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.src_embed = Embedding(src_vocab_size, d_model, max_seq_len, dropout)
        self.tgt_embed = Embedding(tgt_vocab_size, d_model, max_seq_len, dropout)
        
        # Encoder
        encoder_layer = EncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)
        
        # Decoder
        decoder_layer = DecoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)
        
        # Output
        self.generator = Generator(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """ Xavier uniform initialization for parameters """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(output)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
