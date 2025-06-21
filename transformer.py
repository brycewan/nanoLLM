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