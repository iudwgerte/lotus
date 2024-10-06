from transformers import PretrainedConfig

class LotusConfig(PretrainedConfig):
    model_type = "lotus"

    def __init__(self,
                 dim: int = 1536,
                 n_layers: int = 16,
                 n_heads: int = 12,
                 n_kv_heads: int = 12,
                 vocab_size: int = 12800,
                 hidden_dim: int = None,
                 multiple_of: int = 64,
                 norm_eps: float = 1e-5,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 flash_attention: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attention = flash_attention