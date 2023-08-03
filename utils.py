import torch


def copy_parameters(param_official, param_ours):
    """Copy values of one tensor to another.
    Args:
        param_official (torch.Tensor): Tensor to be copied.
        param_ours (torch.Tensor): This tensor will be overwritten in—place with the values from `param_official`
    """
    if param_official.shape != param_ours.shape:
        raise ValueError("`param_official` and `param_ours` must have the same shape.")
    
    with torch.no_grad():
        param_ours.copy_(param_official)

def copy_block(block_official, block_ours):
    """Copy all the parameters within a transformer block.
    Args:
        block_official (torch.nn.Module): Official  transformer block.transformers.models.gpt2.modeling_gpt2.GPT2Block
        block_ours (torch.nn.Module): Our  transformer block.
    """
    b_a = block_official
    b_b = block_ours

    #LN1
    copy_parameters(b_a.ln_1.weight, b_b.ln_1.weight)
    copy_parameters(b_a.ln_1.bias, b_b.ln_1.bias)

    #Attention
    copy_parameters(b_a.attn.c_attn.weight.T, b_b.attention.in_proj_weight)
    copy_parameters(b_a.attn.c_attn.bias, b_b.attention.in_proj_bias)

    copy_parameters(b_a.attn.c_proj.weight.T, b_b.attention.out_proj.weight)
    copy_parameters(b_a.attn.c_proj.bias, b_b.attention.out_proj.bias)

    #LN2
    copy_parameters(b_a.ln_2.weight, b_b.ln_2.weight)
    copy_parameters(b_a.ln_2.bias, b_b.ln_2.bias)

    #MLP
    copy_parameters(b_a.mlp.c_fc.weight.T, b_b.mlp[0].weight)
    copy_parameters(b_a.mlp.c_fc.bias, b_b.mlp[0].bias)

    copy_parameters(b_a.mlp.c_proj.weight.T, b_b.mlp[2].weight)
    copy_parameters(b_a.mlp.c_proj.bias, b_b.mlp[2].bias)

def copy_model(model_official, model_ours):
    """Copy all the parameters within a transformer model.
    Args:
        model_official (torch.nn.Module): Official  transformer model.transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel
        model_ours (torch.nn.Module): GPT
    """

    m_a = model_official
    m_b = model_ours

    #Tokens and Positional Embeddings
    copy_parameters(m_a.transformer.wpe.weight, m_b.pos_emb.weight)
    copy_parameters(m_a.transformer.wte.weight, m_b.token_emb.weight)

    #Block
    for block_official, block_ours in zip(m_a.transformer.h, m_b.blocks):
        copy_block(block_official, block_ours)

    #Head
    copy_parameters(m_a.transformer.ln_f.weight, m_b.ln.weight)
    copy_parameters(m_a.transformer.ln_f.bias, m_b.ln.bias)
    copy_parameters(m_a.lm_head.weight, m_b.head.weight)

@torch.no_grad()
def generate_token(model, token_ixs, temperature = 1.0, sample= False, top_k = None):
    """Generate a new token given a previous token
    Parameters
    ----------
    model:GPT
        Our gpt model.
    token_ixs: list
        List of conditional input token ids.
    temperature: float
        Temperature parameter.
    sample: bool
        If True, we sample from distribution( there is randomness). If false then we take the argmax (there is no randomness)
    top_k: int or None
        If not None then we modify the distribution to only contain the top k logits.
    Returns
    -------
    new_token_ix: int
        Index of the new token.
    """
    context_token_ixs = token_ixs[-model.n_positions :]
    ixs = torch.tensor(context_token_ixs).to(dtype=torch.long)[None, :]
    logits_all = model(ixs)
    logits = logits_all[0, -1, :]
    logits = logits / temperature

    if top_k is not None:
        top_values, _ = torch.topk(logits, k=top_k)
        logits[logits < top_values.min()] = -torch.inf
    
    probs = torch.nn.functional.softmax(logits, dim=0)

    if sample:
        new_token_ix = torch.multinomial(probs, num_samples=1)
    else:
        new_token_ix = probs.argmax()

    return new_token_ix.item()