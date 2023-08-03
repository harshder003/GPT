import torch
import torch.nn as nn
from transformers.activations import gelu_new

class CustomGELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Run forward pass."""
        return gelu_new(x)
    
class Block(nn.Module):
    """Decoder block
    
    Parameters:
    ===========
    n_embed : int
        Dimensionality of the embeddings
    
    n_head : int
        Number of attention heads
    
    n_positions : int 
        Max no. of tokens
    
    attn_pdrop : float
        Probability of dropping the attention weights
    
    resid_pdrop : float
        Probability of dropping after applying MLP
        
    layer_norm_epsilon : float
        Epsilon value for layer normalization
        
    Attributes
    ----------
    ln_1, ln_2: nn.LayerNorm
        Layer normalization
        
    attention : nn.MultiHeadAttention
        Attention Module
        
    mlp : nn.Sequential
        Muli layer Perceptron"""
    
    def __init__(self, *, n_embd, n_head, n_positions, attn_pdrop, resid_pdrop, layer_norm_epsilon):
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)

        self.attention = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=attn_pdrop, bias=True, batch_first=True)
        self.register_buffer("mask",(1- torch.tril(torch.ones(n_positions, n_positions))).to(dtype=torch.bool),)
        
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4*n_embd),
                                 CustomGELU(),
                                 nn.Linear(4*n_embd, n_embd),
                                 nn.Dropout(resid_pdrop))
        
    def forward(self, x):
        """Run forward pass.
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_tokens, n_embed)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_tokens, n_embed)
        """
        batch_size, n_tokens, n_embed = x.shape

        x_ = self.ln_1(x)
        mask = self.mask[:n_tokens,:n_tokens]

        attn_out = self.attention(x_, x_, x_, attn_mask=mask, need_weights=False)[0]
        x = x + attn_out[0]
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    """Entire GPT Module.
    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary
        
    n_layers : int
        Number of decoder blocks to include
        
    n_embed : int
        Number of attention heads
        
    n_positions : int
        Maximum number of tokens.
    
    attn_pdrop : float
        Probability of dropping the attention weights
        
    resid_pdrop : float
        Probability of dropping after applying MLP
        
    layer_norm_epsilon : float
        Hyperparameter for layer normalization
        
    Attributes
    ----------
    token_emb : nn.Embedding
        Token embeddings.
        
    pos_emb : nn.Embedding
        Positional embeddings.
        
    drop : nn.Dropout
        Dropout module to be applied to the embeddings.
        
    blocks : nn.Sequential
        List of decoder blocks.
        
    ln : nn.LayerNorm
        Layer norm applied before applying 'head'
        
    head : nn.Linear
        Final linear layer.
    """

    def __init__ (self, *, vocab_size, n_layer, n_embd, n_head, n_positions, attn_pdrop, embd_pdrop, resid_pdrop, layer_norm_epsilon,):
        super().__init__()

        self.n_positions = n_positions
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(n_positions, n_embd)
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(*[
            Block(n_embd=n_embd, n_head=n_head, n_positions=n_positions, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, layer_norm_epsilon=layer_norm_epsilon)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        """Run forward pass.
        Parameters
        ----------
        idx : torch.Tensor
            Integer tensor of shape (batch_size, n_tokens) where each element is in the range [0,vocab_size)
        Returns
        -------
        logits : torch.Tensor
            Output tensor of shape (batch_size, n_tokens, vocab_size)
        """
        batch_size, n_tokens = idx.shape
        device = idx.device

        if n_tokens > self.n_positions:
            raise ValueError("Too many tokens.")
        
        positions = torch.arange(n_tokens, device=device)
        token_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(positions)[None,...] # (1, n_tokens, n_embed)
        x = self.drop(token_emb + pos_emb) # (batch_size, n_tokens, n_embed)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits