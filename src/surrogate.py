"""
surrogate.py
Differentiable surrogate network for f_eq(T).
"""

import numpy as np
import torch
import torch.nn as nn
from .fd_reference import f_ortho_eq_numpy

class FEqSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, T):
        return self.net(T)

def build_feq_surrogate(device="cpu", epochs=1500):
    T_grid = np.linspace(1.0, 300.0, 2000).astype(np.float32)
    f_grid = np.array([f_ortho_eq_numpy(Ti) for Ti in T_grid], dtype=np.float32)

    X = torch.tensor(T_grid[:,None], dtype=torch.float32, device=device)
    Y = torch.tensor(f_grid[:,None], dtype=torch.float32, device=device)

    sur = FEqSurrogate().to(device)
    opt = torch.optim.Adam(sur.parameters(), lr=1e-3)
    for i in range(epochs):
        opt.zero_grad()
        y = sur(X)
        loss = ((y - Y)**2).mean()
        loss.backward(); opt.step()
        if i % 300 == 0:
            print(f"[surrogate] iter {i:4d}, loss={loss.item():.3e}")
    for p in sur.parameters(): p.requires_grad = False
    return sur
