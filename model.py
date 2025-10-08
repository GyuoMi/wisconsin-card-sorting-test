import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model ,num_heads):
        super().__init__()
        # TODO: weight matrices for Q, K, V and final linear layer
        pass

    def forward(self, x, mask=None):
        # TODO: multi-headed attention logic
        return None
    
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # TODO: two layers for FFN
        pass

    def forward(self, x):
        # TODO: feed forward logic
        return None

class TransformerBlock(nn.Module):
    def __init__(self, d_model ,num_heads):
        super().__init__()
        # TODO: multihead attention, ff layer and layer norms
        pass

    def forward(self, x, mask=None):
        # TODO: forward pass for one transformer block
        # self-attention -> add&norm -> feed-forward -> add&norm
        return None

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_block, num_heads):
        super().__init__()
        # TODO: token and positional embedding layers
        # TODO: stack of TransformerBlock layers
        # TODO: linear layer to map to covab size

    def forward(self, x, mask=None):
        # TODO: full forward pass for transformer model
        return None
        
