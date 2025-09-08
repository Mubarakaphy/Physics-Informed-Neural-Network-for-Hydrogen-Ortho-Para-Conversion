"""
pinn_heat_kinetics.py
Main PINN training script for heat + ortho/para kinetics.
"""

import os, warnings, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from .fd_reference import generate_synthetic
from .surrogate import build_feq_surrogate

# CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
device = torch.device("cpu")

# --- Physical params ---
L = 0.1; t_max = 60.0
alpha = 1e-5; rho_cp = 1.0e6; DeltaH = 1e5
k0 = 1e-3; E_act = 5.0

# Surrogate
sur = build_feq_surrogate(device=device, epochs=1500)
def f_eq_torch(T): return sur(T)

# PINN model
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.act = nn.Tanh()
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        y = x
        for layer in self.net[:-1]:
            y = self.act(layer(y))
        return self.net[-1](y)

# Normalization
x_min, x_max = 0.0, L
t_min, t_max_local = 0.0, t_max
sx = 2.0 / (x_max - x_min)
st = 2.0 / (t_max_local - t_min)
def normalize_X_np(X):
    Xn = X.copy()
    Xn[:,0] = 2*(X[:,0] - x_min)/(x_max - x_min) - 1.0
    Xn[:,1] = 2*(X[:,1] - t_min)/(t_max_local - t_min) - 1.0
    return Xn

# Training routine (simplified)
def train_pinn(num_epochs=1000, collocation_N=2000):
    # FD reference
    x_grid, t_grid, T_all, f_all = generate_synthetic(
        L=L, Nx=81, t_max=t_max, Nt=301,
        alpha=alpha, rho_cp=rho_cp, DeltaH=DeltaH,
        k0=k0, E_act=E_act
    )

    # Collocation points
    x_coll = np.random.rand(collocation_N)*L
    t_coll = np.random.rand(collocation_N)*t_max
    X_coll = np.vstack([x_coll, t_coll]).T
    X_coll_n = normalize_X_np(X_coll)
    X_coll_t = torch.tensor(X_coll_n, dtype=torch.float32, device=device, requires_grad=True)

    # Model
    layers = [2, 64, 64, 64, 2]
    model = PINN(layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    # Training loop (just PDE residuals for brevity)
    for epoch in range(num_epochs):
        model.train(); opt.zero_grad()
        pred = model(X_coll_t); T_c = pred[:,0:1]; f_c = pred[:,1:2]

        # Derivatives
        gT = torch.autograd.grad(T_c, X_coll_t, torch.ones_like(T_c), create_graph=True)[0]
        T_xn, T_tn = gT[:,0:1], gT[:,1:2]
        T_xxn = torch.autograd.grad(T_xn, X_coll_t, torch.ones_like(T_xn), create_graph=True)[0][:,0:1]
        T_t = T_tn*st; T_xx = T_xxn*(sx**2)
        gf = torch.autograd.grad(f_c, X_coll_t, torch.ones_like(f_c), create_graph=True)[0]
        f_t = gf[:,1:2]*st

        T_clamp = T_c.clamp(min=1e-3)
        f_eq_vals = f_eq_torch(T_clamp)
        k_vals = k0 * torch.exp(-E_act / (T_clamp + 1e-8))
        S_c = -DeltaH / rho_cp * f_t

        r_T = T_t - alpha*T_xx - S_c
        r_f = f_t + k_vals*(f_c - f_eq_vals)
        loss = mse(r_T, torch.zeros_like(r_T)) + mse(r_f, torch.zeros_like(r_f))

        loss.backward(); opt.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: Loss {loss.item():.3e}")

    return model, x_grid, t_grid, T_all, f_all

if __name__ == "__main__":
    model, x_grid, t_grid, T_all, f_all = train_pinn()
    print("Training complete")
