import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model ,num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_models must be divis by num_heads"
        # https://logangraves.com/building-a-transformer-2024

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # dim of each head's k/q/v

        # linear layers for q, k, v and final output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_prod_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # the mask is a tensor of 0s and 1s, we want to set the masked pos to a very small number
            # so they become 0 after softmax
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs


    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        # project input into q, k, v
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # reshape qkv to split into multiple heads
        # new shape to be transposed for attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, attn_probs = self.scaled_dot_prod_attention(Q, K, V, mask)

        # now combines heads back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

        # and pass it through the final linear layer
        return self.w_o(attn_output), attn_probs

    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048): # d_ff is usually 4 * d_model
        super().__init__()
        # a simple two layer feed-forward network
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(0.1)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # standard FFN logic: linear -> relu -> dropout -> linear
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model ,num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # one block of the transformer
        # attn -> add&norm -> feed-forward -> add&norm

        # self-attention part
        attn_output, self.attention_weights = self.attention(x, mask)
        # add & norm (skip/residual connection)
        x = self.norm1(x + self.dropout(attn_output))

        # feed-forward part
        ff_ouput = self.ff(x)
        # another add&norm
        x = self.norm2(x + self.dropout(attn_output))

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=256):
        super().__init__()
        # https://medium.com/@laoluoyefolu/transformers-a-practical-guide-with-pytorch-9243b4dc4c37

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # add positional encoding to the input tensor
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_blocks, num_heads, max_seq_length=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # a stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range (num_blocks)])

        # final linear layer to map back to vocab size
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x, mask=None):
        # full forward pass for the transformer 
        # embed tokens, add pos info, then through the blocks
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        # grab the attn weights from the very last block
        # the TransformerBlock conveniently stores them for us
        last_attention_weights = self.blocks[-1].attention_weights

        return self.fc_out(x), last_attention_weights

