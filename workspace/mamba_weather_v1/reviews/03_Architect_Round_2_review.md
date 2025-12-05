# 🏗️ 深度架构蓝图: 03_Architect_Round_2

## 1. 项目概览
- **Project Name**: `Spectral Adaptive Mamba (SAMba)`
- **Style**: Non-Stationary State Space Model with Adaptive Parameterization

### 📦 Dependencies (requirements.txt)
```text
torch>=2.0.0
numpy>=1.21.0
pywavelets>=1.4.0
einops>=0.7.0
```

### ⚙️ Configuration
| Hyperparameter | Default Value |
| :--- | :--- |
| `d_model` | `Hidden dimension size (default: 512)` |
| `d_state` | `State dimension for SSM (default: 16)` |
| `n_basis` | `Number of temporal basis functions (default: 8)` |
| `n_scales` | `Number of wavelet decomposition scales (default: 4)` |
| `n_experts` | `Number of spectral experts (default: 4)` |
| `expert_dim` | `Dimension per expert (default: 128)` |
| `dt_rank` | `Rank for discretization step projection (default: 16)` |
| `n_layers` | `Number of SAMba blocks (default: 6)` |
| `learning_rate` | `Initial learning rate (default: 1e-3)` |
| `seq_len` | `Input sequence length (default: 96)` |
| `pred_len` | `Prediction horizon length (default: 24)` |
| `orthogonal_constraint` | `Method for basis matrix constraints ('cayley'|'householder')` |

---

## 2. 核心文件结构 (Detailed Specs)

### 📄 `src\models\sam.py`
> *Core SAMba model implementation with time-varying state space parameters*

**Imports**: 
`import torch, import torch.nn as nn, import torch.nn.functional as F, from einops import rearrange, repeat, import math`

#### Classes
**`class TemporalBasisParameterization(nn.Module)`**
- *Attributes*: `self.d_model = config.d_model, self.d_state = config.d_state, self.n_basis = config.n_basis, self.basis_matrices = nn.Parameter(torch.randn(n_basis, d_state, d_state)), self.basis_weights = nn.Linear(d_model, n_basis), self.orthogonal_constraint = config.orthogonal_constraint`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: dict -> None` | <ul><li>`super().__init__()`</li><li>`self.d_model = config.d_model`</li><li>`self.d_state = config.d_state`</li><li>`self.n_basis = config.n_basis`</li><li>`self.basis_matrices = nn.Parameter(torch.randn(n_basis, d_state, d_state))`</li><li>`self.basis_weights = nn.Linear(d_model, n_basis)`</li><li>`self.orthogonal_constraint = config.orthogonal_constraint`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`B, L, D = x.shape  # [batch, seq_len, d_model]`</li><li>`alpha = self.basis_weights(x)  # [B, L, n_basis]`</li><li>`alpha = F.softmax(alpha, dim=-1)`</li><li>`basis_matrices = self._apply_orthogonal_constraint(self.basis_matrices)`</li><li>`A_t = torch.einsum('blk,kij->blij', alpha, basis_matrices)  # [B, L, d_state, d_state]`</li><li>`return A_t`</li></ul> |
| **_apply_orthogonal_constraint** | `matrices: torch.Tensor -> torch.Tensor` | <ul><li>`if self.orthogonal_constraint == 'cayley':`</li><li>`    return self._cayley_transform(matrices)`</li><li>`elif self.orthogonal_constraint == 'householder':`</li><li>`    return self._householder_projection(matrices)`</li><li>`else:`</li><li>`    return matrices`</li></ul> |

**`class MultiScaleStationarization(nn.Module)`**
- *Attributes*: `self.d_model = config.d_model, self.n_scales = config.n_scales, self.wavelet_type = config.wavelet_type, self.conv_filters = self._build_wavelet_filters(), self.residual_proj = nn.Linear(d_model * n_scales, d_model)`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: dict -> None` | <ul><li>`super().__init__()`</li><li>`self.d_model = config.d_model`</li><li>`self.n_scales = config.n_scales`</li><li>`self.wavelet_type = config.wavelet_type`</li><li>`self.conv_filters = self._build_wavelet_filters()`</li><li>`self.residual_proj = nn.Linear(d_model * n_scales, d_model)`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`B, L, D = x.shape  # [batch, seq_len, d_model]`</li><li>`x_perm = x.permute(0, 2, 1)  # [B, D, L]`</li><li>`wavelet_coeffs = []`</li><li>`for scale in range(self.n_scales):`</li><li>`    filter_length = min(2**(scale+1), L)`</li><li>`    conv_filter = self.conv_filters[scale]`</li><li>`    padding = (filter_length - 1) // 2`</li><li>`    coeff = F.conv1d(x_perm, conv_filter, padding=padding, groups=D)`</li><li>`    wavelet_coeffs.append(coeff)`</li><li>`wavelet_components = torch.stack(wavelet_coeffs, dim=1)  # [B, n_scales, D, L]`</li><li>`wavelet_components = wavelet_components.permute(0, 3, 1, 2)  # [B, L, n_scales, D]`</li><li>`reconstructed = self.residual_proj(wavelet_components.reshape(B, L, -1))`</li><li>`return reconstructed`</li></ul> |

**`class LinearSpectralGating(nn.Module)`**
- *Attributes*: `self.d_model = config.d_model, self.n_experts = config.n_experts, self.expert_dim = config.expert_dim, self.query_proj = nn.Linear(d_model, n_experts), self.key_proj = nn.Linear(d_model, n_experts), self.value_proj = nn.Linear(d_model, expert_dim * n_experts), self.output_proj = nn.Linear(expert_dim * n_experts, d_model)`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: dict -> None` | <ul><li>`super().__init__()`</li><li>`self.d_model = config.d_model`</li><li>`self.n_experts = config.n_experts`</li><li>`self.expert_dim = config.expert_dim`</li><li>`self.query_proj = nn.Linear(d_model, n_experts)`</li><li>`self.key_proj = nn.Linear(d_model, n_experts)`</li><li>`self.value_proj = nn.Linear(d_model, expert_dim * n_experts)`</li><li>`self.output_proj = nn.Linear(expert_dim * n_experts, d_model)`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`B, L, D = x.shape  # [batch, seq_len, d_model]`</li><li>`Q = self.query_proj(x)  # [B, L, n_experts]`</li><li>`K = self.key_proj(x)  # [B, L, n_experts]`</li><li>`V = self.value_proj(x)  # [B, L, expert_dim * n_experts]`</li><li>`gating_weights = torch.einsum('ble,ble->be', Q, K) / math.sqrt(L)  # [B, n_experts]`</li><li>`gating_weights = F.softmax(gating_weights, dim=-1)`</li><li>`V_reshaped = V.reshape(B, L, self.n_experts, self.expert_dim)`</li><li>`expert_output = torch.einsum('be,bleo->blo', gating_weights, V_reshaped)`</li><li>`expert_output = expert_output.reshape(B, L, -1)`</li><li>`output = self.output_proj(expert_output)`</li><li>`return output`</li></ul> |

**`class AdaptiveSelectiveScanning(nn.Module)`**
- *Attributes*: `self.d_model = config.d_model, self.d_state = config.d_state, self.dt_rank = config.dt_rank, self.dt_proj = nn.Linear(dt_rank, d_model), self.A_parameterization = TemporalBasisParameterization(config), self.B_proj = nn.Linear(d_model, d_state, bias=False), self.C_proj = nn.Linear(d_model, d_state, bias=False), self.D_proj = nn.Linear(d_model, d_model, bias=False)`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: dict -> None` | <ul><li>`super().__init__()`</li><li>`self.d_model = config.d_model`</li><li>`self.d_state = config.d_state`</li><li>`self.dt_rank = config.dt_rank`</li><li>`self.dt_proj = nn.Linear(dt_rank, d_model)`</li><li>`self.A_parameterization = TemporalBasisParameterization(config)`</li><li>`self.B_proj = nn.Linear(d_model, d_state, bias=False)`</li><li>`self.C_proj = nn.Linear(d_model, d_state, bias=False)`</li><li>`self.D_proj = nn.Linear(d_model, d_model, bias=False)`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`B, L, D = x.shape  # [batch, seq_len, d_model]`</li><li>`A_t = self.A_parameterization(x)  # [B, L, d_state, d_state]`</li><li>`B_t = self.B_proj(x)  # [B, L, d_state]`</li><li>`C_t = self.C_proj(x)  # [B, L, d_state]`</li><li>`dt = F.softplus(self.dt_proj(x[:, :, :self.dt_rank]))  # [B, L, d_model]`</li><li>`h = torch.zeros(B, self.d_state, device=x.device)  # [B, d_state]`</li><li>`outputs = []`</li><li>`for t in range(L):`</li><li>`    A_discrete = torch.matrix_exp(A_t[:, t] * dt[:, t].unsqueeze(-1).unsqueeze(-1))`</li><li>`    h = torch.einsum('bij,bj->bi', A_discrete, h) + B_t[:, t] * x[:, t]`</li><li>`    y_t = torch.einsum('bi,bi->b', C_t[:, t], h) + self.D_proj(x[:, t])`</li><li>`    outputs.append(y_t.unsqueeze(1))`</li><li>`output = torch.cat(outputs, dim=1)  # [B, L, D]`</li><li>`return output`</li></ul> |

**`class SAMbaBlock(nn.Module)`**
- *Attributes*: `self.d_model = config.d_model, self.norm1 = nn.LayerNorm(d_model), self.norm2 = nn.LayerNorm(d_model), self.stationarization = MultiScaleStationarization(config), self.spectral_gating = LinearSpectralGating(config), self.selective_scan = AdaptiveSelectiveScanning(config), self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: dict -> None` | <ul><li>`super().__init__()`</li><li>`self.d_model = config.d_model`</li><li>`self.norm1 = nn.LayerNorm(d_model)`</li><li>`self.norm2 = nn.LayerNorm(d_model)`</li><li>`self.stationarization = MultiScaleStationarization(config)`</li><li>`self.spectral_gating = LinearSpectralGating(config)`</li><li>`self.selective_scan = AdaptiveSelectiveScanning(config)`</li><li>`self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`B, L, D = x.shape  # [batch, seq_len, d_model]`</li><li>`x_stationary = self.stationarization(self.norm1(x))`</li><li>`x_gated = self.spectral_gating(x_stationary)`</li><li>`x_ssm = self.selective_scan(x_gated)`</li><li>`x = x + x_ssm`</li><li>`x = x + self.mlp(self.norm2(x))`</li><li>`return x`</li></ul> |

**`class SAMba(nn.Module)`**
- *Attributes*: `self.d_model = config.d_model, self.n_layers = config.n_layers, self.input_proj = nn.Linear(config.input_dim, d_model), self.output_proj = nn.Linear(d_model, config.output_dim), self.blocks = nn.ModuleList([SAMbaBlock(config) for _ in range(n_layers)])`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: dict -> None` | <ul><li>`super().__init__()`</li><li>`self.d_model = config.d_model`</li><li>`self.n_layers = config.n_layers`</li><li>`self.input_proj = nn.Linear(config.input_dim, d_model)`</li><li>`self.output_proj = nn.Linear(d_model, config.output_dim)`</li><li>`self.blocks = nn.ModuleList([SAMbaBlock(config) for _ in range(n_layers)])`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`B, L, D_in = x.shape  # [batch, seq_len, input_dim]`</li><li>`x = self.input_proj(x)  # [B, L, d_model]`</li><li>`for block in self.blocks:`</li><li>`    x = block(x)`</li><li>`output = self.output_proj(x)  # [B, L, output_dim]`</li><li>`return output`</li></ul> |



---
### 📄 `src\data\time_series_loader.py`
> *Data loader for time series forecasting tasks*

**Imports**: 
`import torch, from torch.utils.data import Dataset, DataLoader, import numpy as np`

#### Classes
**`class TimeSeriesDataset(Dataset)`**
- *Attributes*: `self.data = data, self.seq_len = seq_len, self.pred_len = pred_len`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `data: torch.Tensor, seq_len: int, pred_len: int -> None` | <ul><li>`super().__init__()`</li><li>`self.data = data`</li><li>`self.seq_len = seq_len`</li><li>`self.pred_len = pred_len`</li></ul> |
| **__len__** | ` -> int` | <ul><li>`return len(self.data) - self.seq_len - self.pred_len + 1`</li></ul> |
| **__getitem__** | `idx: int -> tuple[torch.Tensor, torch.Tensor]` | <ul><li>`x = self.data[idx:idx+self.seq_len]  # [seq_len, input_dim]`</li><li>`y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]  # [pred_len, output_dim]`</li><li>`return x, y`</li></ul> |



---
### 📄 `src\training\trainer.py`
> *Training loop and evaluation for SAMba model*

**Imports**: 
`import torch, import torch.nn as nn, from torch.optim import AdamW, import numpy as np, from tqdm import tqdm`

#### Classes
**`class SAMbaTrainer(nn.Module)`**
- *Attributes*: `self.model = model, self.config = config, self.optimizer = AdamW(model.parameters(), lr=config.learning_rate), self.criterion = nn.MSELoss()`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `model: nn.Module, config: dict -> None` | <ul><li>`super().__init__()`</li><li>`self.model = model`</li><li>`self.config = config`</li><li>`self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)`</li><li>`self.criterion = nn.MSELoss()`</li></ul> |
| **train_epoch** | `train_loader: DataLoader -> float` | <ul><li>`self.model.train()`</li><li>`total_loss = 0`</li><li>`for batch_idx, (x, y) in enumerate(train_loader):`</li><li>`    self.optimizer.zero_grad()`</li><li>`    pred = self.model(x)`</li><li>`    loss = self.criterion(pred, y)`</li><li>`    loss.backward()`</li><li>`    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)`</li><li>`    self.optimizer.step()`</li><li>`    total_loss += loss.item()`</li><li>`return total_loss / len(train_loader)`</li></ul> |
| **validate** | `val_loader: DataLoader -> float` | <ul><li>`self.model.eval()`</li><li>`total_loss = 0`</li><li>`with torch.no_grad():`</li><li>`    for x, y in val_loader:`</li><li>`        pred = self.model(x)`</li><li>`        loss = self.criterion(pred, y)`</li><li>`        total_loss += loss.item()`</li><li>`return total_loss / len(val_loader)`</li></ul> |



---
### 📄 `src\tests\test_components.py`
> *Unit tests for SAMba components*

**Imports**: 
`import torch, import torch.nn as nn, import pytest, from ..models.sam import TemporalBasisParameterization, MultiScaleStationarization, LinearSpectralGating, AdaptiveSelectiveScanning`


#### Functions
- 🔹 **`test_temporal_basis_gradients`**
  - Logic:
1. `config = {'d_model': 64, 'd_state': 16, 'n_basis': 8, 'orthogonal_constraint': 'cayley'}`
1. `module = TemporalBasisParameterization(config)`
1. `x = torch.randn(2, 32, 64, requires_grad=True)`
1. `A_t = module(x)`
1. `loss = A_t.sum()`
1. `loss.backward()`
1. `assert x.grad is not None`
1. `assert not torch.isnan(x.grad).any()`
- 🔹 **`test_wavelet_differentiability`**
  - Logic:
1. `config = {'d_model': 64, 'n_scales': 4, 'wavelet_type': 'db4'}`
1. `module = MultiScaleStationarization(config)`
1. `x = torch.randn(2, 32, 64, requires_grad=True)`
1. `output = module(x)`
1. `loss = output.sum()`
1. `loss.backward()`
1. `assert x.grad is not None`
1. `assert not torch.isnan(x.grad).any()`
- 🔹 **`test_linear_spectral_gating_complexity`**
  - Logic:
1. `config = {'d_model': 64, 'n_experts': 4, 'expert_dim': 32}`
1. `module = LinearSpectralGating(config)`
1. `x = torch.randn(2, 1024, 64)`
1. `import time`
1. `start = time.time()`
1. `output = module(x)`
1. `end = time.time()`
1. `assert (end - start) < 1.0  # Should be fast for long sequences`

---
### 📄 `configs\default_config.py`
> *Default configuration for SAMba model*

**Imports**: 
``



---

## 3. 主流程 (Execution Flow)
1. Load time series data -> 2. Apply multi-scale stationarization -> 3. Route through spectral gating -> 4. Process with adaptive selective scanning -> 5. Output forecast

---
<!-- SYSTEM SEPARATOR -->

# 🟢 用户决策区

**决策 (Action)**: [ APPROVE ] 

**反馈意见 (Feedback)**:
<!-- 比如：forward 函数里的 shape 好像不对，应该是 [B, D, L] -->