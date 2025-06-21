import torch
import torch.nn as nn
import torch.nn.functional as F

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