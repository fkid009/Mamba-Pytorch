import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def selective_scan(
        x,      # (B, L, D): input
        delta,  # (B, L, D): time step per token
        A,      # (D, N): fixed parameter
        B,      # (B, L, N): 저장 기준
        C,      # (B, L, N): 읽기 기준
        D,      # (D,): skip-connection
):  # learnable parameter가 없어 함수로 구현
    batch, L, D_in = x.shape
    N = A.shape[1]

    h = torch.zeros(batch, D_in, N, device=x.device, dtype=x.dtype)
    ys = []

    # ponytail: 순차 for-loop scan이라 L에 비례해 느림. 필요하면 parallel scan(mamba-ssm CUDA kernel)으로 교체
    for t in range(L):
        delta_t = delta[:, t].unsqueeze(-1)  # (B, D, 1)

        # discretization
        dA = torch.exp(delta_t * A)          # (B, D, N)
        dB = delta_t * B[:, t].unsqueeze(1)  # (B, D, N)

        # state update
        h = dA * h + dB * x[:, t].unsqueeze(-1)  # (B, D, N)

        # read
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)  # (B, L, D)
    return y + x * D


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model  # 내부 차원 확장
        self.d_state = d_state           # 채널당 상태 크기
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # depthwise causal conv1d (kernel_size = d_conv)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)  # (D, N)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def ssm(self, x):
        A = -torch.exp(self.A_log)  # 음수로 유지해 상태가 감쇠하도록

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))  # (B, L, D)

        return selective_scan(x, delta, A, B, C, self.D)

    def forward(self, x):
        _, L, _ = x.shape

        # input
        x, z = self.in_proj(x).chunk(2, dim=-1)  # 각각 (B, L, D_INNER)

        # causal conv1d: 오른쪽 padding을 잘라 미래 정보 차단
        x = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x = F.silu(x)

        # selective ssm
        y = self.ssm(x)

        # gating
        y = y * F.silu(z)
        return self.out_proj(y)  # (B, L, D_MODEL)
