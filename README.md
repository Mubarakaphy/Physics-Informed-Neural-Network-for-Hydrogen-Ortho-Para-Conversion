# 🔥 Physics-Informed Neural Network for Hydrogen Ortho/Para Conversion

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This project demonstrates how **Physics-Informed Neural Networks (PINNs)** can solve coupled **heat conduction + ortho/para hydrogen conversion kinetics**, an important problem in **hydrogen storage and cryogenic systems**.  

It compares PINN predictions with a **finite-difference (FD)** solver and shows how physics-informed ML can model complex thermochemical systems where **data are scarce but physical laws are known**.

---

## ✨ Features

- 🧪 **Finite-Difference Solver (FD)**  
  Stable explicit scheme with CFL condition auto-adjustment.

- 🤖 **PINN Implementation (PyTorch)**  
  - Fully differentiable physics loss (heat + kinetics PDEs)  
  - Input normalization & gradient-based residuals  
  - Adam → LBFGS refinement for stable convergence  

- 📐 **Surrogate for Statistical Mechanics**  
  - Differentiable neural surrogate for the equilibrium ortho-fraction \( f_{\rm eq}(T) \)  
  - Trained from statistical mechanics partition function  
  - Ensures PINN uses *the same physics* as FD reference

- 📊 **Visualization**  
  - Centerline temperature & ortho-fraction vs time  
  - Residual heatmaps (where the PINN struggles most)  

---

## 🚀 Getting Started

Clone the repo:
```bash
git clone https://github.com/Mubarakaphy/pinn-heat-kinetics.git
cd pinn-heat-kinetics
