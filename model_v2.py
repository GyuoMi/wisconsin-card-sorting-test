import torch
import torch.nn as nn
import math

# This file implements a GPT-2 style, Pre-Norm transformer 
# block similar to a demo done by Andrew Karpasky
# which is generally more stable to train.

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model # <-- This is the fix
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs) # Dropout on attention probs
        
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def forward(self, x_norm, mask=None):
        # x_norm is the *already normalised* input
        batch_size, seq_length, _ = x_norm.size()

        Q = self.w_q(x_norm)
        K = self.w_k(x_norm)
        V = self.w_v(x_norm)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
        # Project and apply final dropout
        return self.proj_dropout(self.w_o(attn_output)), attn_probs

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4 # Standard 4x expansion
            
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU() # Using ReLU
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) # Add dropout

    def forward(self, x_norm):
        # x_norm is the *already normalised* input
        x = self.linear_1(x_norm)
        x = self.activation(x)
        x = self.linear_2(x)
        return self.dropout(x) # Apply dropout

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        
        # --- This is the Pre-Norm architecture ---
        self.norm1 = nn.LayerNorm(d_model) # Normalise *before* attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model) # Normalise *before* feed-forward
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        self.attention_weights = None # For analysis

    def forward(self, x, mask=None):
        
        # --- Pre-Norm Attention ---
        # 1. Normalise
        # 2. Pass through attention
        # 3. Add to original 'x' (residual connection)
        attn_output, self.attention_weights = self.attention(self.norm1(x), mask)
        x = x + attn_output
        
        # --- Pre-Norm Feed-Forward ---
        # 1. Normalise
        # 2. Pass through feed-forward
        # 3. Add to the result of the previous step
        x = x + self.ff(self.norm2(x))
        
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_blocks, num_heads, max_seq_length, d_ff=None, dropout=0.1):
        super().__init__()
        
        # Compositional Embedding logic
        # use colours, 4 shapes, 4 quantities
        self.colour_embed = nn.Embedding(4, d_model)
        self.shape_embed = nn.Embedding(4, d_model)
        self.quantity_embed = nn.Embedding(4, d_model)
        
        # We have 6 special tokens (C1, C2, C3, C4, SEP, EOS)
        # These correspond to token IDs 64-69
        self.special_embed = nn.Embedding(6, d_model)
        # ---------------------------------------------------
        
        self.positional_encoding = nn.Embedding(max_seq_length, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_blocks)]
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size) # Output is still 70 classes
        self.max_seq_length = max_seq_length

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # A standard initialisation strategy
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        assert seq_len <= self.max_seq_length, "Input sequence is longer than model's max_seq_length"

        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0) # (1, T)
        
        # New Embedding Forward Pass (ABSOLUTELY CRUCIAL AFFFF)
        # Create masks to separate cards (0-63) from specials (64-69)
        card_mask = (x < 64).float().unsqueeze(-1)
        special_mask = (x >= 64).float().unsqueeze(-1)
        
        # 1. Card Embeddings (Compositional)
        # We use modular arithmetic to find the feature indices from the token ID
        quant_idx = (x % 4).long()
        shape_idx = ((x // 4) % 4).long()
        colour_idx = ((x // 16) % 4).long()
        
        c_emb = self.colour_embed(colour_idx)
        s_emb = self.shape_embed(shape_idx)
        q_emb = self.quantity_embed(quant_idx)
        
        # Sum the features and apply the card mask
        # quite literally telling the model the card's vec 
        # that it's a sum of the parts
        card_emb = (c_emb + s_emb + q_emb) * card_mask

        # 2. Special Embeddings
        # We map tokens 64-69 to indices 0-5
        special_idx = (x - 64).long()
        # We clamp the indices to handle the out-of-bounds -64 for card tokens
        special_emb = self.special_embed(torch.clamp(special_idx, 0, 5)) * special_mask
        
        # 3. Combine
        tok_emb = card_emb + special_emb
        # ----------------------------------
        
        pos_emb = self.positional_encoding(positions)
        
        x = self.emb_dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.final_norm(x)
        
        logits = self.fc_out(x)
        
        # Get attention weights from the last block for analysis
        last_attention_weights = self.blocks[-1].attention_weights
        
        return logits, last_attention_weights

