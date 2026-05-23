"""
CLOVER 组件: ClinicalMLP (3→64→256)。
将 [PSA, Volume, has_cancer] 编码为临床特征 embedding，
用于对比学习中的临床-视觉对齐。
"""

import torch
import torch.nn as nn


class ClinicalMLP(nn.Module):
    """将临床参数向量映射到对比 embedding 空间。"""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
