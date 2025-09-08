"""
fd_reference.py
Finite-difference (FD) solver for 1D heat + ortho/para kinetics.
"""

import numpy as np

# --- Physical constants ---
kB = 1.380649e-23  # J/K
Theta_rot = 85.4   # K

def E_J(J):
    return kB * Theta_rot * J * (J + 1)

def f_ortho_eq_numpy(T, Jmax=40):
    """Equilibrium ortho fraction from rotational partition function."""
    T = max(float(T), 1.0)
    Js = np.arange(0, Jmax+1)
    energies = np.exp(-np.array([E_J(J) for J in Js]) / (kB*T + 1e-30))
    g_ns = np.where(Js % 2 == 0, 1, 3)   # para even=1, ortho odd=3
    degeneracy = (2*Js + 1) * g_ns
    Z = np.sum(degeneracy * energies)
    ortho_sum = np.sum(degeneracy[Js % 2 == 1] * energies[Js % 2 == 1])
    return float(ortho_sum / Z)

def generate_synthetic(L=0.1, Nx=81, t_max=60.0, Nt=301,
                       alpha=1e-5, rho_cp=1e6, DeltaH=1e5,
                       k0=1e-3, E_act=5.0):
    """
    Explicit FD solver with CFL stability adjustment.
    Returns x, t, T_all, f_all.
    """
    dx = L/(Nx-1)
    x = np.linspace(0, L, Nx)

    max_dt = 0.5 * dx * dx / alpha
    dt_user = t_max/(Nt-1)
    if dt_user > max_dt:
        dt = max_dt
        Nt = int(np.ceil(t_max / dt)) + 1
        dt = t_max / (Nt - 1)
        print(f"[FD] Adjusted Nt={Nt}, dt={dt:.3e} (max_dt {max_dt:.3e})")
    else:
        dt = dt_user

    t = np.linspace(0, t_max, Nt)

    # Initial conditions
    T = np.ones(Nx) * 20.0
    f = np.ones(Nx) * 0.75
    T += 5.0 * np.exp(-((x - L/2)**2)/(2*(0.01)**2))  # Gaussian bump

    T_all = np.zeros((Nt, Nx)); T_all[0,:] = T.copy()
    f_all = np.zeros((Nt, Nx)); f_all[0,:] = f.copy()

    for n in range(1, Nt):
        T_for = np.maximum(T, 1e-3)
        k_vals = k0 * np.exp(-E_act / (T_for + 1e-12))
        f_eq_vals = np.array([f_ortho_eq_numpy(Ti) for Ti in np.maximum(T, 1.0)])
        dfdt = -k_vals * (f - f_eq_vals)
        f_new = f + dt * dfdt

        S = -DeltaH / rho_cp * dfdt
        T_new = T.copy()
        T_new[1:-1] = T[1:-1] + alpha * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2]) + dt * S[1:-1]
        T_new[0] = 20.0; T_new[-1] = 20.0

        if (np.isnan(T_new).any() or np.isnan(f_new).any()):
            print(f"[FD] NaN at step {n}, truncating")
            T_all = T_all[:n,:]; f_all = f_all[:n,:]; t = t[:n]; break

        T = np.maximum(T_new, 1e-6)
        f = np.clip(f_new, 0.0, 1.0)
        T_all[n,:] = T; f_all[n,:] = f

    return x, t, T_all, f_all
