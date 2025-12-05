# 🏗️ 深度架构蓝图: 03_Architect_Round_1

## 1. 项目概览
- **Project Name**: `PhysMamba`
- **Style**: Physics-Constrained State Space Model for Weather Forecasting

### 📦 Dependencies (requirements.txt)
```text
torch>=2.0
numpy
scipy
```

### ⚙️ Configuration
| Hyperparameter | Default Value |
| :--- | :--- |
| `lr` | `1e-4` |
| `lambda_mass` | `0.1` |
| `lambda_energy` | `0.1` |
| `lambda_momentum` | `0.1` |
| `gamma_mass` | `1.0` |
| `gamma_energy` | `1.0` |
| `gamma_momentum` | `1.0` |
| `tau` | `0.1` |
| `sigma` | `0.05` |

---

## 2. 核心文件结构 (Detailed Specs)

### 📄 `src\models\phys_mamba.py`
> *Core PhysMamba architecture with physics-constrained state space models*

**Imports**: 
`torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math`

#### Classes
**`class PhysicsConstrainedParameterization(nn.Module)`**
- *Attributes*: `self.d_model, self.state_dim, self.lambda_mass, self.lambda_energy, self.lambda_momentum, self.A_base, self.A_mass, self.A_energy, self.A_momentum`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, d_model: int, state_dim: int, lambda_mass: float = 0.1, lambda_energy: float = 0.1, lambda_momentum: float = 0.1 -> None` | <ul><li>`self.d_model = d_model`</li><li>`self.state_dim = state_dim`</li><li>`self.lambda_mass = lambda_mass`</li><li>`self.lambda_energy = lambda_energy`</li><li>`self.lambda_momentum = lambda_momentum`</li><li>`self.A_base = nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)`</li><li>`self.A_mass = self._create_mass_conservation_matrix()`</li><li>`self.A_energy = self._create_energy_conservation_matrix()`</li><li>`self.A_momentum = self._create_momentum_conservation_matrix()`</li></ul> |
| **_create_mass_conservation_matrix** | `self -> torch.Tensor` | <ul><li>`matrix = torch.zeros(self.state_dim, self.state_dim)`</li><li>`# Enforce mass conservation: sum of flows = 0`</li><li>`for i in range(self.state_dim):`</li><li>`    for j in range(self.state_dim):`</li><li>`        if i != j:`</li><li>`            matrix[i, j] = 1.0 / (self.state_dim - 1)`</li><li>`            matrix[i, i] = -1.0`</li><li>`return matrix`</li></ul> |
| **_create_energy_conservation_matrix** | `self -> torch.Tensor` | <ul><li>`matrix = torch.zeros(self.state_dim, self.state_dim)`</li><li>`# Energy conservation: symmetric constraints`</li><li>`for i in range(self.state_dim):`</li><li>`    for j in range(i + 1, self.state_dim):`</li><li>`        matrix[i, j] = 0.5`</li><li>`        matrix[j, i] = 0.5`</li><li>`        matrix[i, i] = -0.5`</li><li>`        matrix[j, j] = -0.5`</li><li>`return matrix`</li></ul> |
| **_create_momentum_conservation_matrix** | `self -> torch.Tensor` | <ul><li>`matrix = torch.zeros(self.state_dim, self.state_dim)`</li><li>`# Momentum conservation: anti-symmetric constraints`</li><li>`for i in range(self.state_dim):`</li><li>`    for j in range(self.state_dim):`</li><li>`        if i < j:`</li><li>`            matrix[i, j] = 1.0`</li><li>`            matrix[j, i] = -1.0`</li><li>`return matrix`</li></ul> |
| **forward** | `self -> torch.Tensor` | <ul><li>`A_physics = (self.lambda_mass * self.A_mass + `</li><li>`                  self.lambda_energy * self.A_energy + `</li><li>`                  self.lambda_momentum * self.A_momentum)`</li><li>`A_total = self.A_base + A_physics`</li><li>`return A_total`</li></ul> |

**`class RegimeAwareDiscretization(nn.Module)`**
- *Attributes*: `self.tau, self.sigma, self.ou_process, self.delta_min, self.delta_max`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, tau: float = 0.1, sigma: float = 0.05, delta_min: float = 0.01, delta_max: float = 1.0 -> None` | <ul><li>`self.tau = tau`</li><li>`self.sigma = sigma`</li><li>`self.ou_process = OrnsteinUhlenbeckProcess(tau, sigma)`</li><li>`self.delta_min = delta_min`</li><li>`self.delta_max = delta_max`</li></ul> |
| **forward** | `self, A: torch.Tensor, seq_len: int -> torch.Tensor` | <ul><li>`# Sample regime variations from OU process`</li><li>`eta_t = self.ou_process.sample(seq_len)`</li><li>`# Compute adaptive time steps`</li><li>`delta_t = self.tau + self.sigma * eta_t`</li><li>`delta_t = torch.clamp(delta_t, self.delta_min, self.delta_max)`</li><li>`# Discretize A matrix`</li><li>`A_bar = torch.matrix_exp(delta_t.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(0))`</li><li>`return A_bar`</li></ul> |

**`class MultiScaleStateEvolution(nn.Module)`**
- *Attributes*: `self.state_dim, self.d_model, self.B_proj, self.C_proj, self.D`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, d_model: int, state_dim: int -> None` | <ul><li>`self.state_dim = state_dim`</li><li>`self.d_model = d_model`</li><li>`self.B_proj = nn.Linear(d_model, state_dim)`</li><li>`self.C_proj = nn.Linear(state_dim, d_model)`</li><li>`self.D = nn.Parameter(torch.ones(d_model))`</li></ul> |
| **forward** | `self, x: torch.Tensor, A_bar: torch.Tensor, h_prev: torch.Tensor = None -> Tuple[torch.Tensor, torch.Tensor]` | <ul><li>`batch_size, seq_len, _ = x.shape`</li><li>`if h_prev is None:`</li><li>`    h_prev = torch.zeros(batch_size, self.state_dim, device=x.device)`</li><li>`# Project input to state dimension`</li><li>`B_bar = self.B_proj(x)`</li><li>`# Perform state evolution`</li><li>`h_t = torch.einsum('bli,bij->blj', A_bar, h_prev.unsqueeze(1)) + B_bar`</li><li>`# Project state back to output dimension`</li><li>`y_t = torch.einsum('bli,ij->blj', h_t, self.C_proj.weight) + self.D * x`</li><li>`return y_t, h_t[:, -1]`</li></ul> |

**`class PhysMambaBlock(nn.Module)`**
- *Attributes*: `self.d_model, self.state_dim, self.phys_param, self.regime_disc, self.state_evol`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, d_model: int, state_dim: int, lambda_mass: float = 0.1, lambda_energy: float = 0.1, lambda_momentum: float = 0.1 -> None` | <ul><li>`self.d_model = d_model`</li><li>`self.state_dim = state_dim`</li><li>`self.phys_param = PhysicsConstrainedParameterization(d_model, state_dim, lambda_mass, lambda_energy, lambda_momentum)`</li><li>`self.regime_disc = RegimeAwareDiscretization()`</li><li>`self.state_evol = MultiScaleStateEvolution(d_model, state_dim)`</li></ul> |
| **forward** | `self, x: torch.Tensor, h_prev: torch.Tensor = None -> Tuple[torch.Tensor, torch.Tensor]` | <ul><li>`# Get physics-constrained A matrix`</li><li>`A = self.phys_param()`</li><li>`# Discretize with regime awareness`</li><li>`A_bar = self.regime_disc(A, x.shape[1])`</li><li>`# Perform state evolution`</li><li>`y, h_next = self.state_evol(x, A_bar, h_prev)`</li><li>`return y, h_next`</li></ul> |



---
### 📄 `src\models\processes.py`
> *Stochastic processes for regime modeling*

**Imports**: 
`torch, numpy as np`

#### Classes
**`class OrnsteinUhlenbeckProcess(nn.Module)`**
- *Attributes*: `self.tau, self.sigma, self.theta, self.current_value`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, tau: float, sigma: float, theta: float = 1.0 -> None` | <ul><li>`self.tau = tau`</li><li>`self.sigma = sigma`</li><li>`self.theta = theta`</li><li>`self.current_value = torch.tensor(0.0)`</li></ul> |
| **sample** | `self, n_samples: int -> torch.Tensor` | <ul><li>`samples = []`</li><li>`current = self.current_value`</li><li>`for _ in range(n_samples):`</li><li>`    # OU process update: dx = theta * (mu - x) * dt + sigma * dW`</li><li>`    dw = torch.randn_like(current) * torch.sqrt(torch.tensor(self.tau))`</li><li>`    current = current + self.theta * (-current) * self.tau + self.sigma * dw`</li><li>`    samples.append(current)`</li><li>`return torch.stack(samples)`</li></ul> |
| **reset** | `self -> None` | <ul><li>`self.current_value = torch.tensor(0.0)`</li></ul> |



---
### 📄 `src\models\conservation.py`
> *Physics conservation law constraints and loss computation*

**Imports**: 
`torch, torch.nn as nn`

#### Classes
**`class ConservationConstraints(nn.Module)`**
- *Attributes*: `self.gamma_mass, self.gamma_energy, self.gamma_momentum`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, gamma_mass: float = 1.0, gamma_energy: float = 1.0, gamma_momentum: float = 1.0 -> None` | <ul><li>`self.gamma_mass = gamma_mass`</li><li>`self.gamma_energy = gamma_energy`</li><li>`self.gamma_momentum = gamma_momentum`</li></ul> |
| **mass_conservation** | `self, h_t: torch.Tensor -> torch.Tensor` | <ul><li>`# Mass conservation: total mass should be preserved`</li><li>`mass_initial = h_t[:, 0].sum(dim=-1)`</li><li>`mass_final = h_t[:, -1].sum(dim=-1)`</li><li>`violation = torch.abs(mass_final - mass_initial)`</li><li>`return violation`</li></ul> |
| **energy_conservation** | `self, h_t: torch.Tensor -> torch.Tensor` | <ul><li>`# Energy conservation: sum of squares should be preserved`</li><li>`energy_initial = (h_t[:, 0] ** 2).sum(dim=-1)`</li><li>`energy_final = (h_t[:, -1] ** 2).sum(dim=-1)`</li><li>`violation = torch.abs(energy_final - energy_initial)`</li><li>`return violation`</li></ul> |
| **momentum_conservation** | `self, h_t: torch.Tensor -> torch.Tensor` | <ul><li>`# Momentum conservation: weighted sum should be preserved`</li><li>`weights = torch.arange(h_t.shape[-1], device=h_t.device).float()`</li><li>`momentum_initial = (h_t[:, 0] * weights).sum(dim=-1)`</li><li>`momentum_final = (h_t[:, -1] * weights).sum(dim=-1)`</li><li>`violation = torch.abs(momentum_final - momentum_initial)`</li><li>`return violation`</li></ul> |
| **forward** | `self, h_t: torch.Tensor -> torch.Tensor` | <ul><li>`mass_loss = self.gamma_mass * self.mass_conservation(h_t)`</li><li>`energy_loss = self.gamma_energy * self.energy_conservation(h_t)`</li><li>`momentum_loss = self.gamma_momentum * self.momentum_conservation(h_t)`</li><li>`total_loss = mass_loss + energy_loss + momentum_loss`</li><li>`return total_loss`</li></ul> |



---
### 📄 `src\training\loss.py`
> *Physics-regularized loss functions*

**Imports**: 
`torch, torch.nn as nn, torch.nn.functional as F`

#### Classes
**`class PhysicsRegularizedLoss(nn.Module)`**
- *Attributes*: `self.mse_loss, self.conservation_constraints`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, gamma_mass: float = 1.0, gamma_energy: float = 1.0, gamma_momentum: float = 1.0 -> None` | <ul><li>`self.mse_loss = nn.MSELoss()`</li><li>`self.conservation_constraints = ConservationConstraints(gamma_mass, gamma_energy, gamma_momentum)`</li></ul> |
| **forward** | `self, pred: torch.Tensor, target: torch.Tensor, h_t: torch.Tensor -> torch.Tensor` | <ul><li>`mse = self.mse_loss(pred, target)`</li><li>`conservation_loss = self.conservation_constraints(h_t)`</li><li>`total_loss = mse + conservation_loss`</li><li>`return total_loss`</li></ul> |



---
### 📄 `src\main.py`
> *Main training and evaluation script*

**Imports**: 
`torch, torch.nn as nn, torch.optim as optim, numpy as np, os, sys, from models.phys_mamba import PhysMambaBlock, from training.loss import PhysicsRegularizedLoss`


#### Functions
- 🔹 **`train_phys_mamba`**
  - Logic:
1. `optimizer = optim.Adam(model.parameters(), lr=lr)`
1. `criterion = PhysicsRegularizedLoss()`
1. `train_losses = []`
1. `val_losses = []`
1. `for epoch in range(num_epochs):`
1. `    model.train()`
1. `    epoch_loss = 0.0`
1. `    for batch_idx, (data, target) in enumerate(train_loader):`
1. `        optimizer.zero_grad()`
1. `        output, h_t = model(data)`
1. `        loss = criterion(output, target, h_t)`
1. `        loss.backward()`
1. `        optimizer.step()`
1. `        epoch_loss += loss.item()`
1. `    train_losses.append(epoch_loss / len(train_loader))`
1. `    # Validation`
1. `    model.eval()`
1. `    val_loss = 0.0`
1. `    with torch.no_grad():`
1. `        for data, target in val_loader:`
1. `            output, h_t = model(data)`
1. `            loss = criterion(output, target, h_t)`
1. `            val_loss += loss.item()`
1. `    val_losses.append(val_loss / len(val_loader))`
1. `return {'train': train_losses, 'val': val_losses}`

---

## 3. 主流程 (Execution Flow)
Data loading -> Physics parameter encoding -> Regime-aware discretization -> Multi-scale state evolution -> Physics-constrained output -> Loss computation with conservation regularization

---
<!-- SYSTEM SEPARATOR -->

# 🟢 用户决策区

**决策 (Action)**: [ APPROVE ] 

**反馈意见 (Feedback)**:
<!-- 比如：forward 函数里的 shape 好像不对，应该是 [B, D, L] -->