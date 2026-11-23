import numpy as np

def token_embedding_lookup(vocab_size: int, embed_dim: int, token_ids: list, seed: int = 0) -> list:
    """
    Build a random embedding table of shape (vocab_size, embed_dim) using
    np.random.default_rng(seed).standard_normal(...), then return the rows
    corresponding to token_ids as a nested list.
    """
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((vocab_size, embed_dim))
    # print(E.shape)     # should print (5, 3)
    # print(E[0])        # should start with 0.1257..., -0.1321..., 0.6404...
    return E[token_ids].tolist()