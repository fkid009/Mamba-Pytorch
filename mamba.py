import torch.nn as nn

from mamba_block import MambaBlock


class ResidualBlock(nn.Module):
    """
    x + Mamba(RMSNorm(x))
    """
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=1e-5)
        self.mixer = MambaBlock(d_model, d_state, d_conv, expand)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class Mamba(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)  # 순서는 recurrence가 처리하므로 pos_emb 없음

        self.layers = nn.ModuleList([
            ResidualBlock(d_model, d_state, d_conv, expand) for _ in range(n_layer)
        ])

        self.norm_f = nn.RMSNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        # x: (B, L) 정수 텐서
        x = self.token_emb(x)

        for layer in self.layers:
            x = layer(x)

        return self.head(self.norm_f(x))  # (B, L, VOCAB_SIZE)
