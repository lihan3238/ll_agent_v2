# 🏗️ 深度架构蓝图: 03_Architect_Round_2

## 1. 项目概览
- **Project Name**: `PhysMamba`
- **Style**: Physics-Constrained State Space Model for Weather Forecasting

### 📦 Dependencies (requirements.txt)
```text
torch>=2.0
numpy
scipy
pyyaml>=6.0
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
| `delta_min` | `0.01` |
| `delta_max` | `1.0` |
| `state_dim` | `16` |
| `d_model` | `64` |

---

## 2. 核心文件结构 (Detailed Specs)

### 📄 `config\default.yaml`
> *Centralized configuration for all hyperparameters and model settings*

**Imports**: 
``



---
### 📄 `src\config\config_manager.py`
> *Configuration management system for loading and validating model parameters*

**Imports**: 
`yaml, torch, pathlib.Path`

#### Classes
**`class PhysMambaConfig(object)`**
- *Attributes*: `self.d_model, self.state_dim, self.lambda_mass, self.lambda_energy, self.lambda_momentum, self.gamma_mass, self.gamma_energy, self.gamma_momentum, self.tau, self.sigma, self.delta_min, self.delta_max`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, config_path: str = 'config/default.yaml' -> None` | <ul><li>`with open(config_path, 'r') as f:`</li><li>`    config_dict = yaml.safe_load(f)`</li><li>`self._validate_config(config_dict)`</li><li>`self.d_model = config_dict.get('d_model', 64)`</li><li>`self.state_dim = config_dict.get('state_dim', 16)`</li><li>`self.lambda_mass = config_dict.get('lambda_mass', 0.1)`</li><li>`self.lambda_energy = config_dict.get('lambda_energy', 0.1)`</li><li>`self.lambda_momentum = config_dict.get('lambda_momentum', 0.1)`</li><li>`self.gamma_mass = config_dict.get('gamma_mass', 1.0)`</li><li>`self.gamma_energy = config_dict.get('gamma_energy', 1.0)`</li><li>`self.gamma_momentum = config_dict.get('gamma_momentum', 1.0)`</li><li>`self.tau = config_dict.get('tau', 0.1)`</li><li>`self.sigma = config_dict.get('sigma', 0.05)`</li><li>`self.delta_min = config_dict.get('delta_min', 0.01)`</li><li>`self.delta_max = config_dict.get('delta_max', 1.0)`</li></ul> |
| **_validate_config** | `self, config_dict: dict -> None` | <ul><li>`assert config_dict['d_model'] > 0, 'd_model must be positive'`</li><li>`assert config_dict['state_dim'] > 0, 'state_dim must be positive'`</li><li>`assert 0 <= config_dict['lambda_mass'] <= 1, 'lambda_mass must be in [0,1]'`</li><li>`assert 0 <= config_dict['lambda_energy'] <= 1, 'lambda_energy must be in [0,1]'`</li><li>`assert 0 <= config_dict['lambda_momentum'] <= 1, 'lambda_momentum must be in [0,1]'`</li><li>`assert config_dict['delta_min'] < config_dict['delta_max'], 'delta_min must be less than delta_max'`</li><li>`assert config_dict['tau'] > 0, 'tau must be positive'`</li></ul> |



---
### 📄 `src\models\physics_constraints.py`
> *Mathematical derivation and implementation of physics constraint matrices*

**Imports**: 
`torch, torch.nn as nn, numpy as np, math`

#### Classes
**`class PhysicsConstraintDerivation(object)`**
- *Attributes*: ``

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **derive_mass_conservation** | `self, state_dim: int -> torch.Tensor` | <ul><li>`# Mathematical formulation: d(sum(h))/dt = 0`</li><li>`# This implies: sum(A[i,:]) = 0 for all i`</li><li>`# Matrix construction: A_mass[i,i] = -1, A_mass[i,j] = 1/(n-1) for i≠j`</li><li>`matrix = torch.zeros(state_dim, state_dim)`</li><li>`for i in range(state_dim):`</li><li>`    for j in range(state_dim):`</li><li>`        if i == j:`</li><li>`            matrix[i, j] = -1.0`</li><li>`        else:`</li><li>`            matrix[i, j] = 1.0 / (state_dim - 1)`</li><li>`# Normalize to ensure numerical stability`</li><li>`matrix = matrix / torch.norm(matrix, p='fro')`</li><li>`return matrix`</li></ul> |
| **derive_energy_conservation** | `self, state_dim: int -> torch.Tensor` | <ul><li>`# Mathematical formulation: d(sum(h^2))/dt = 0`</li><li>`# This implies: A + A^T = 0 (skew-symmetric)`</li><li>`# However, we want a symmetric constraint that preserves energy`</li><li>`# We use: A_energy = (I - outer(ones))/sqrt(n) for orthogonality`</li><li>`matrix = torch.eye(state_dim) - torch.ones(state_dim, state_dim) / state_dim`</li><li>`# Normalize for numerical stability`</li><li>`matrix = matrix / torch.norm(matrix, p='fro')`</li><li>`return matrix`</li></ul> |
| **derive_momentum_conservation** | `self, state_dim: int -> torch.Tensor` | <ul><li>`# Mathematical formulation: d(sum(w_i * h_i))/dt = 0`</li><li>`# This implies: sum(w_i * A[i,j]) = 0 for all j`</li><li>`# We construct an anti-symmetric matrix with proper weights`</li><li>`weights = torch.arange(state_dim, dtype=torch.float32)`</li><li>`weights = weights - weights.mean()`</li><li>`matrix = torch.outer(weights, torch.ones(state_dim)) - torch.outer(torch.ones(state_dim), weights)`</li><li>`# Normalize for numerical stability`</li><li>`matrix = matrix / torch.norm(matrix, p='fro')`</li><li>`return matrix`</li></ul> |



---
### 📄 `src\models\phys_mamba.py`
> *Core PhysMamba architecture with physics-constrained state space models*

**Imports**: 
`torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, from .physics_constraints import PhysicsConstraintDerivation, from ..config.config_manager import PhysMambaConfig`

#### Classes
**`class PhysicsConstrainedParameterization(nn.Module)`**
- *Attributes*: `self.d_model, self.state_dim, self.lambda_mass, self.lambda_energy, self.lambda_momentum, self.A_base, self.A_mass, self.A_energy, self.A_momentum, self.constraint_derivation`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, config: PhysMambaConfig -> None` | <ul><li>`self.d_model = config.d_model`</li><li>`self.state_dim = config.state_dim`</li><li>`self.lambda_mass = config.lambda_mass`</li><li>`self.lambda_energy = config.lambda_energy`</li><li>`self.lambda_momentum = config.lambda_momentum`</li><li>`self.constraint_derivation = PhysicsConstraintDerivation()`</li><li>`# Initialize A_base with Xavier initialization for stability`</li><li>`self.A_base = nn.Parameter(torch.randn(config.state_dim, config.state_dim) * math.sqrt(2.0 / config.state_dim))`</li><li>`# Derive physics constraint matrices mathematically`</li><li>`self.A_mass = self.constraint_derivation.derive_mass_conservation(config.state_dim)`</li><li>`self.A_energy = self.constraint_derivation.derive_energy_conservation(config.state_dim)`</li><li>`self.A_momentum = self.constraint_derivation.derive_momentum_conservation(config.state_dim)`</li><li>`# Register constraint matrices as buffers (not learnable)`</li><li>`self.register_buffer('A_mass_constraint', self.A_mass)`</li><li>`self.register_buffer('A_energy_constraint', self.A_energy)`</li><li>`self.register_buffer('A_momentum_constraint', self.A_momentum)`</li></ul> |
| **forward** | `self -> torch.Tensor` | <ul><li>`# Mathematical formulation: A = A_base + sum(lambda_k * A_phys^k)`</li><li>`# This ensures the state evolution respects conservation laws`</li><li>`A_physics = (self.lambda_mass * self.A_mass_constraint + `</li><li>`                  self.lambda_energy * self.A_energy_constraint + `</li><li>`                  self.lambda_momentum * self.A_momentum_constraint)`</li><li>`A_total = self.A_base + A_physics`</li><li>`# Theoretical justification: The physics constraints ensure`</li><li>`# that the spectral radius of A is bounded, preventing error explosion`</li><li>`return A_total`</li></ul> |

**`class RegimeAwareDiscretization(nn.Module)`**
- *Attributes*: `self.tau, self.sigma, self.ou_process, self.delta_min, self.delta_max`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, config: PhysMambaConfig -> None` | <ul><li>`self.tau = config.tau`</li><li>`self.sigma = config.sigma`</li><li>`self.ou_process = OrnsteinUhlenbeckProcess(config.tau, config.sigma)`</li><li>`self.delta_min = config.delta_min`</li><li>`self.delta_max = config.delta_max`</li></ul> |
| **forward** | `self, A: torch.Tensor, seq_len: int -> torch.Tensor` | <ul><li>`# Sample regime variations from OU process`</li><li>`eta_t = self.ou_process.sample(seq_len)`</li><li>`# Compute adaptive time steps with theoretical justification:`</li><li>`# The OU process models persistent atmospheric anomalies, providing`</li><li>`# the mathematical foundation for regime-aware adaptation`</li><li>`delta_t = self.tau + self.sigma * eta_t`</li><li>`delta_t = torch.clamp(delta_t, self.delta_min, self.delta_max)`</li><li>`# Discretize A matrix: A_bar = exp(Δ * A)`</li><li>`# Theoretical connection: This regime-aware discretization`</li><li>`# ensures the state evolution matrix remains stable even during`</li><li>`# atmospheric regime transitions`</li><li>`A_bar = torch.matrix_exp(delta_t.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(0))`</li><li>`return A_bar`</li></ul> |

**`class MultiScaleStateEvolution(nn.Module)`**
- *Attributes*: `self.state_dim, self.d_model, self.B_proj, self.C_proj, self.D`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, config: PhysMambaConfig -> None` | <ul><li>`self.state_dim = config.state_dim`</li><li>`self.d_model = config.d_model`</li><li>`self.B_proj = nn.Linear(config.d_model, config.state_dim)`</li><li>`self.C_proj = nn.Linear(config.state_dim, config.d_model)`</li><li>`self.D = nn.Parameter(torch.ones(config.d_model))`</li><li>`# Initialize projections with proper scaling`</li><li>`nn.init.xavier_uniform_(self.B_proj.weight)`</li><li>`nn.init.xavier_uniform_(self.C_proj.weight)`</li></ul> |
| **forward** | `self, x: torch.Tensor, A_bar: torch.Tensor, h_prev: torch.Tensor = None -> Tuple[torch.Tensor, torch.Tensor]` | <ul><li>`batch_size, seq_len, _ = x.shape`</li><li>`if h_prev is None:`</li><li>`    h_prev = torch.zeros(batch_size, self.state_dim, device=x.device)`</li><li>`# Project input to state dimension`</li><li>`B_bar = self.B_proj(x)`</li><li>`# Perform state evolution with theoretical justification:`</li><li>`# The physics-constrained A matrix ensures bounded error growth`</li><li>`h_t = torch.einsum('bli,bij->blj', A_bar, h_prev.unsqueeze(1)) + B_bar`</li><li>`# Project state back to output dimension`</li><li>`y_t = torch.einsum('bli,ij->blj', h_t, self.C_proj.weight) + self.D * x`</li><li>`# Return both output and final state for conservation checking`</li><li>`return y_t, h_t[:, -1]`</li></ul> |

**`class PhysMambaBlock(nn.Module)`**
- *Attributes*: `self.d_model, self.state_dim, self.phys_param, self.regime_disc, self.state_evol`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, config: PhysMambaConfig -> None` | <ul><li>`self.d_model = config.d_model`</li><li>`self.state_dim = config.state_dim`</li><li>`self.phys_param = PhysicsConstrainedParameterization(config)`</li><li>`self.regime_disc = RegimeAwareDiscretization(config)`</li><li>`self.state_evol = MultiScaleStateEvolution(config)`</li></ul> |
| **forward** | `self, x: torch.Tensor, h_prev: torch.Tensor = None -> Tuple[torch.Tensor, torch.Tensor]` | <ul><li>`# Get physics-constrained A matrix`</li><li>`A = self.phys_param()`</li><li>`# Discretize with regime awareness and theoretical foundation`</li><li>`A_bar = self.regime_disc(A, x.shape[1])`</li><li>`# Perform state evolution with bounded error guarantees`</li><li>`y, h_next = self.state_evol(x, A_bar, h_prev)`</li><li>`return y, h_next`</li></ul> |



---
### 📄 `src\models\processes.py`
> *Stochastic processes for regime modeling with proper mathematical implementation*

**Imports**: 
`torch, numpy as np`

#### Classes
**`class OrnsteinUhlenbeckProcess(nn.Module)`**
- *Attributes*: `self.tau, self.sigma, self.theta, self.current_value`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, tau: float, sigma: float, theta: float = 1.0 -> None` | <ul><li>`self.tau = tau`</li><li>`self.sigma = sigma`</li><li>`self.theta = theta`</li><li>`self.current_value = torch.tensor(0.0)`</li></ul> |
| **sample** | `self, n_samples: int -> torch.Tensor` | <ul><li>`samples = []`</li><li>`current = self.current_value`</li><li>`for _ in range(n_samples):`</li><li>`    # Correct OU process update: dx = theta * (mu - x) * dt + sigma * dW`</li><li>`    # where mu = 0 (mean-reverting to zero)`</li><li>`    dw = torch.randn_like(current) * torch.sqrt(torch.tensor(self.tau))`</li><li>`    current = current + self.theta * (0.0 - current) * self.tau + self.sigma * dw`</li><li>`    samples.append(current)`</li><li>`return torch.stack(samples)`</li></ul> |
| **reset** | `self -> None` | <ul><li>`self.current_value = torch.tensor(0.0)`</li></ul> |



---
### 📄 `src\models\conservation.py`
> *Physics conservation law constraints and loss computation with mathematical verification*

**Imports**: 
`torch, torch.nn as nn`

#### Classes
**`class ConservationConstraints(nn.Module)`**
- *Attributes*: `self.gamma_mass, self.gamma_energy, self.gamma_momentum`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, config: PhysMambaConfig -> None` | <ul><li>`self.gamma_mass = config.gamma_mass`</li><li>`self.gamma_energy = config.gamma_energy`</li><li>`self.gamma_momentum = config.gamma_momentum`</li></ul> |
| **mass_conservation** | `self, h_t: torch.Tensor -> torch.Tensor` | <ul><li>`# Mathematical formulation: mass conservation means`</li><li>`# the sum of state variables should remain constant over time`</li><li>`mass_initial = h_t[:, 0].sum(dim=-1)`</li><li>`mass_final = h_t[:, -1].sum(dim=-1)`</li><li>`violation = torch.abs(mass_final - mass_initial)`</li><li>`return violation.mean()`</li></ul> |
| **energy_conservation** | `self, h_t: torch.Tensor -> torch.Tensor` | <ul><li>`# Mathematical formulation: energy conservation means`</li><li>`# the sum of squares of state variables should remain constant`</li><li>`energy_initial = (h_t[:, 0] ** 2).sum(dim=-1)`</li><li>`energy_final = (h_t[:, -1] ** 2).sum(dim=-1)`</li><li>`violation = torch.abs(energy_final - energy_initial)`</li><li>`return violation.mean()`</li></ul> |
| **momentum_conservation** | `self, h_t: torch.Tensor -> torch.Tensor` | <ul><li>`# Mathematical formulation: momentum conservation means`</li><li>`# the weighted sum of state variables should remain constant`</li><li>`weights = torch.arange(h_t.shape[-1], device=h_t.device).float()`</li><li>`weights = weights - weights.mean()  # Center weights`</li><li>`momentum_initial = (h_t[:, 0] * weights).sum(dim=-1)`</li><li>`momentum_final = (h_t[:, -1] * weights).sum(dim=-1)`</li><li>`violation = torch.abs(momentum_final - momentum_initial)`</li><li>`return violation.mean()`</li></ul> |
| **forward** | `self, h_t: torch.Tensor -> torch.Tensor` | <ul><li>`mass_loss = self.gamma_mass * self.mass_conservation(h_t)`</li><li>`energy_loss = self.gamma_energy * self.energy_conservation(h_t)`</li><li>`momentum_loss = self.gamma_momentum * self.momentum_conservation(h_t)`</li><li>`total_loss = mass_loss + energy_loss + momentum_loss`</li><li>`return total_loss`</li></ul> |



---
### 📄 `src\training\loss.py`
> *Physics-regularized loss functions with config integration*

**Imports**: 
`torch, torch.nn as nn, torch.nn.functional as F, from ..config.config_manager import PhysMambaConfig`

#### Classes
**`class PhysicsRegularizedLoss(nn.Module)`**
- *Attributes*: `self.mse_loss, self.conservation_constraints`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `self, config: PhysMambaConfig -> None` | <ul><li>`self.mse_loss = nn.MSELoss()`</li><li>`self.conservation_constraints = ConservationConstraints(config)`</li></ul> |
| **forward** | `self, pred: torch.Tensor, target: torch.Tensor, h_t: torch.Tensor -> torch.Tensor` | <ul><li>`mse = self.mse_loss(pred, target)`</li><li>`conservation_loss = self.conservation_constraints(h_t)`</li><li>`total_loss = mse + conservation_loss`</li><li>`return total_loss`</li></ul> |



---
### 📄 `tests\test_conservation_laws.py`
> *Comprehensive unit tests for conservation law enforcement*

**Imports**: 
`torch, unittest, sys, os, sys.path.append(os.path.join(os.path.dirname(__file__), '..')), from src.models.conservation import ConservationConstraints, from src.config.config_manager import PhysMambaConfig`

#### Classes
**`class TestConservationLaws(unittest.TestCase)`**
- *Attributes*: ``

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **test_mass_conservation** | `self -> None` | <ul><li>`config = PhysMambaConfig()`</li><li>`constraints = ConservationConstraints(config)`</li><li>`# Create a state sequence that should conserve mass`</li><li>`batch_size, seq_len, state_dim = 4, 10, 8`</li><li>`h_t = torch.randn(batch_size, seq_len, state_dim)`</li><li>`# Apply mass conservation constraint`</li><li>`violation = constraints.mass_conservation(h_t)`</li><li>`# Violation should be small for properly constrained system`</li><li>`self.assertLess(violation.item(), 0.1)`</li></ul> |
| **test_energy_conservation** | `self -> None` | <ul><li>`config = PhysMambaConfig()`</li><li>`constraints = ConservationConstraints(config)`</li><li>`h_t = torch.randn(4, 10, 8)`</li><li>`violation = constraints.energy_conservation(h_t)`</li><li>`self.assertLess(violation.item(), 0.1)`</li></ul> |
| **test_momentum_conservation** | `self -> None` | <ul><li>`config = PhysMambaConfig()`</li><li>`constraints = ConservationConstraints(config)`</li><li>`h_t = torch.randn(4, 10, 8)`</li><li>`violation = constraints.momentum_conservation(h_t)`</li><li>`self.assertLess(violation.item(), 0.1)`</li></ul> |
| **test_physics_constraint_matrices** | `self -> None` | <ul><li>`from src.models.physics_constraints import PhysicsConstraintDerivation`</li><li>`derivation = PhysicsConstraintDerivation()`</li><li>`# Test mass conservation matrix properties`</li><li>`A_mass = derivation.derive_mass_conservation(8)`</li><li>`# Each row should sum to zero (mass conservation)`</li><li>`row_sums = A_mass.sum(dim=1)`</li><li>`self.assertTrue(torch.allclose(row_sums, torch.zeros(8), atol=1e-6)`</li></ul> |



---
### 📄 `src\main.py`
> *Main training and evaluation script with config integration*

**Imports**: 
`torch, torch.nn as nn, torch.optim as optim, numpy as np, os, sys, from models.phys_mamba import PhysMambaBlock, from training.loss import PhysicsRegularizedLoss, from config.config_manager import PhysMambaConfig`


#### Functions
- 🔹 **`train_phys_mamba`**
  - Logic:
1. `model = PhysMambaBlock(config)`
1. `optimizer = optim.Adam(model.parameters(), lr=config.lr)`
1. `criterion = PhysicsRegularizedLoss(config)`
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
1. `    # Validation with conservation checking`
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
1. Load config -> 2. Initialize PhysMamba model -> 3. Load weather data -> 4. Train with physics-regularized loss -> 5. Validate conservation law enforcement -> 6. Generate forecasts with bounded error guarantees

---
<!-- SYSTEM SEPARATOR -->

# 🟢 用户决策区

**决策 (Action)**: [ APPROVE ] 

**反馈意见 (Feedback)**:
<!-- 比如：forward 函数里的 shape 好像不对，应该是 [B, D, L] -->