# 🏗️ 深度架构蓝图: 03_Architect_Round_1

## 1. 项目概览
- **Project Name**: `Spectral Adaptive Mamba (SAMba)`
- **Style**: Non-Stationary State Space Models with Adaptive Parameterization

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
| `d_model` | `512 (model dimension)` |
| `d_state` | `128 (state dimension)` |
| `d_conv` | `4 (convolution dimension)` |
| `expand` | `2 (expansion factor)` |
| `n_layers` | `12 (number of SAMba blocks)` |
| `n_basis` | `8 (number of temporal basis functions)` |
| `n_experts` | `4 (number of spectral experts)` |
| `n_scales` | `5 (wavelet decomposition scales)` |
| `dt_min` | `0.001 (minimum discretization step)` |
| `dt_max` | `0.1 (maximum discretization step)` |
| `learning_rate` | `1e-4 (optimizer learning rate)` |
| `batch_size` | `32 (training batch size)` |
| `seq_len` | `96 (input sequence length)` |
| `pred_len` | `24 (prediction length)` |

---

## 2. 核心文件结构 (Detailed Specs)

### 📄 `src\config.py`
> *Configuration management for SAMba hyperparameters*

**Imports**: 
`from dataclasses import dataclass, from typing import Optional, Tuple`

#### Classes
**`class SAMbaConfig()`**
- *Attributes*: `d_model: int = 512, d_state: int = 128, d_conv: int = 4, expand: int = 2, n_layers: int = 12, n_basis: int = 8, n_experts: int = 4, n_scales: int = 5, dt_min: float = 0.001, dt_max: float = 0.1, dt_init: str = 'random', learn_dt: bool = True, activation: str = 'silu', use_spectral_gating: bool = True, use_wavelet_stationarization: bool = True`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__post_init__** | ` -> None` | <ul><li>`assert self.d_model > 0, 'd_model must be positive'`</li><li>`assert self.d_state > 0, 'd_state must be positive'`</li><li>`assert self.n_basis > 0, 'n_basis must be positive'`</li><li>`assert self.n_experts > 0, 'n_experts must be positive'`</li></ul> |



---
### 📄 `src\models\temporal_basis.py`
> *Temporal basis parameterization for time-varying state transitions*

**Imports**: 
`import torch, import torch.nn as nn, import torch.nn.functional as F, from einops import rearrange, repeat`

#### Classes
**`class TemporalBasisParameterization(nn.Module)`**
- *Attributes*: `self.d_state = config.d_state, self.n_basis = config.n_basis, self.d_model = config.d_model, self.basis_matrices = nn.Parameter(torch.randn(n_basis, d_state, d_state)), self.basis_weights = nn.Linear(d_model, n_basis), self.orthogonal_proj = nn.Linear(d_state, d_state)`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: SAMbaConfig -> None` | <ul><li>`super().__init__()`</li><li>`self.d_state = config.d_state`</li><li>`self.n_basis = config.n_basis`</li><li>`self.d_model = config.d_model`</li><li>`self.basis_matrices = nn.Parameter(torch.randn(n_basis, d_state, d_state))`</li><li>`self.basis_weights = nn.Linear(d_model, n_basis)`</li><li>`self.orthogonal_proj = nn.Linear(d_state, d_state)`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`B, L, D = x.shape  # [batch, seq_len, d_model]`</li><li>`alpha = self.basis_weights(x)  # [B, L, n_basis]`</li><li>`alpha = F.softmax(alpha, dim=-1)  # [B, L, n_basis]`</li><li>`basis_expanded = repeat(self.basis_matrices, 'K N M -> B L K N M', B=B, L=L)  # [B, L, K, N, M]`</li><li>`alpha_expanded = repeat(alpha, 'B L K -> B L K N M', N=self.d_state, M=self.d_state)  # [B, L, K, N, M]`</li><li>`A_t = (alpha_expanded * basis_expanded).sum(dim=2)  # [B, L, N, M]`</li><li>`A_t = self.orthogonal_proj(A_t)  # [B, L, N, M]`</li><li>`return A_t  # Time-varying state transition matrices`</li></ul> |



---
### 📄 `src\models\wavelet_stationarization.py`
> *Multi-scale stationarization using wavelet decomposition*

**Imports**: 
`import torch, import torch.nn as nn, import pywt, import numpy as np`

#### Classes
**`class MultiScaleStationarization(nn.Module)`**
- *Attributes*: `self.n_scales = config.n_scales, self.wavelet_type = 'db4', self.scale_projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_scales)]), self.residual_proj = nn.Linear(d_model, d_model)`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: SAMbaConfig -> None` | <ul><li>`super().__init__()`</li><li>`self.n_scales = config.n_scales`</li><li>`self.wavelet_type = 'db4'`</li><li>`self.scale_projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_scales)])`</li><li>`self.residual_proj = nn.Linear(d_model, d_model)`</li></ul> |
| **wavelet_decompose** | `x: torch.Tensor -> Tuple[torch.Tensor, torch.Tensor]` | <ul><li>`B, L, D = x.shape  # [batch, seq_len, d_model]`</li><li>`x_np = x.detach().cpu().numpy()`</li><li>`wavelet_coeffs = []`</li><li>`for b in range(B):`</li><li>`    for d in range(D):`</li><li>`        coeffs = pywt.wavedec(x_np[b, :, d], self.wavelet_type, level=self.n_scales)`</li><li>`        wavelet_coeffs.append(coeffs)`</li><li>`w_s = torch.stack([torch.from_numpy(np.concatenate(level_coeffs)) for level_coeffs in zip(*wavelet_coeffs)])`</li><li>`w_s = w_s.to(x.device)`</li><li>`epsilon = self.residual_proj(x)  # [B, L, D]`</li><li>`return w_s, epsilon  # Multi-scale patterns and residual`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`w_s, epsilon = self.wavelet_decompose(x)  # [B, scales*L, D], [B, L, D]`</li><li>`scale_outputs = []`</li><li>`for i, proj in enumerate(self.scale_projections):`</li><li>`    scale_slice = w_s[:, i*L:(i+1)*L, :]  # [B, L, D]`</li><li>`    scale_out = proj(scale_slice)  # [B, L, D]`</li><li>`    scale_outputs.append(scale_out)`</li><li>`stationarized = torch.stack(scale_outputs, dim=-1).sum(dim=-1)  # [B, L, D]`</li><li>`output = stationarized + epsilon  # [B, L, D]`</li><li>`return output`</li></ul> |



---
### 📄 `src\models\spectral_gating.py`
> *Differentiable spectral gating for expert routing*

**Imports**: 
`import torch, import torch.nn as nn, import torch.nn.functional as F, from einops import rearrange`

#### Classes
**`class SpectralGating(nn.Module)`**
- *Attributes*: `self.n_experts = config.n_experts, self.d_model = config.d_model, self.query_proj = nn.Linear(d_model, d_model), self.key_proj = nn.Linear(d_model, d_model), self.expert_projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: SAMbaConfig -> None` | <ul><li>`super().__init__()`</li><li>`self.n_experts = config.n_experts`</li><li>`self.d_model = config.d_model`</li><li>`self.query_proj = nn.Linear(d_model, d_model)`</li><li>`self.key_proj = nn.Linear(d_model, d_model)`</li><li>`self.expert_projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`B, L, D = x.shape  # [batch, seq_len, d_model]`</li><li>`Q = self.query_proj(x)  # [B, L, D]`</li><li>`K = self.key_proj(x)  # [B, L, D]`</li><li>`g_t = torch.bmm(Q, K.transpose(1, 2)) / (D ** 0.5)  # [B, L, L]`</li><li>`g_t = F.softmax(g_t, dim=-1)  # [B, L, L]`</li><li>`expert_outputs = []`</li><li>`for expert_proj in self.expert_projections:`</li><li>`    expert_out = expert_proj(x)  # [B, L, D]`</li><li>`    expert_outputs.append(expert_out)`</li><li>`expert_stack = torch.stack(expert_outputs, dim=-1)  # [B, L, D, E]`</li><li>`g_t_expanded = repeat(g_t, 'B L1 L2 -> B L1 L2 E', E=self.n_experts)  # [B, L, L, E]`</li><li>`routed_output = torch.einsum('blle,blde->bld', g_t_expanded, expert_stack)  # [B, L, D]`</li><li>`return routed_output  # Gated expert outputs`</li></ul> |



---
### 📄 `src\models\adaptive_scanning.py`
> *Adaptive selective scanning with input-dependent discretization*

**Imports**: 
`import torch, import torch.nn as nn, import torch.nn.functional as F`

#### Classes
**`class AdaptiveSelectiveScanning(nn.Module)`**
- *Attributes*: `self.d_model = config.d_model, self.d_state = config.d_state, self.dt_min = config.dt_min, self.dt_max = config.dt_max, self.dt_proj = nn.Linear(d_model, 1), self.B_proj = nn.Linear(d_model, d_state), self.C_proj = nn.Linear(d_model, d_state), self.D_proj = nn.Linear(d_model, d_model)`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: SAMbaConfig -> None` | <ul><li>`super().__init__()`</li><li>`self.d_model = config.d_model`</li><li>`self.d_state = config.d_state`</li><li>`self.dt_min = config.dt_min`</li><li>`self.dt_max = config.dt_max`</li><li>`self.dt_proj = nn.Linear(d_model, 1)`</li><li>`self.B_proj = nn.Linear(d_model, d_state)`</li><li>`self.C_proj = nn.Linear(d_model, d_state)`</li><li>`self.D_proj = nn.Linear(d_model, d_model)`</li></ul> |
| **discretize** | `A_t: torch.Tensor, B_t: torch.Tensor, delta_t: torch.Tensor -> Tuple[torch.Tensor, torch.Tensor]` | <ul><li>`delta_t = delta_t.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]`</li><li>`A_bar = torch.exp(A_t * delta_t)  # [B, L, N, M]`</li><li>`B_bar = (A_t.inverse() @ (A_bar - torch.eye(self.d_state))) @ B_t.unsqueeze(-1)  # [B, L, N, 1]`</li><li>`B_bar = B_bar.squeeze(-1)  # [B, L, N]`</li><li>`return A_bar, B_bar  # Discretized parameters`</li></ul> |
| **forward** | `x: torch.Tensor, A_t: torch.Tensor -> torch.Tensor` | <ul><li>`B, L, D = x.shape  # [batch, seq_len, d_model]`</li><li>`delta_t = self.dt_proj(x)  # [B, L, 1]`</li><li>`delta_t = F.softplus(delta_t)  # [B, L, 1]`</li><li>`delta_t = torch.clamp(delta_t, self.dt_min, self.dt_max)  # [B, L, 1]`</li><li>`B_t = self.B_proj(x)  # [B, L, N]`</li><li>`C_t = self.C_proj(x)  # [B, L, N]`</li><li>`D_t = self.D_proj(x)  # [B, L, D]`</li><li>`A_bar, B_bar = self.discretize(A_t, B_t, delta_t)  # [B, L, N, M], [B, L, N]`</li><li>`h = torch.zeros(B, self.d_state, device=x.device)  # [B, N]`</li><li>`outputs = []`</li><li>`for t in range(L):`</li><li>`    h = A_bar[:, t] @ h.unsqueeze(-1) + B_bar[:, t].unsqueeze(-1)  # [B, N, 1]`</li><li>`h = h.squeeze(-1)  # [B, N]`</li><li>`y_t = (C_t[:, t] @ h.unsqueeze(-1)).squeeze(-1) + D_t[:, t]  # [B, D]`</li><li>`outputs.append(y_t)`</li><li>`y = torch.stack(outputs, dim=1)  # [B, L, D]`</li><li>`return y  # Output sequence`</li></ul> |



---
### 📄 `src\models\samba_block.py`
> *Core SAMba block integrating all components*

**Imports**: 
`import torch, import torch.nn as nn, from .temporal_basis import TemporalBasisParameterization, from .wavelet_stationarization import MultiScaleStationarization, from .spectral_gating import SpectralGating, from .adaptive_scanning import AdaptiveSelectiveScanning`

#### Classes
**`class SAMbaBlock(nn.Module)`**
- *Attributes*: `self.config = config, self.norm = nn.LayerNorm(config.d_model), self.temporal_basis = TemporalBasisParameterization(config), self.wavelet_stationarization = MultiScaleStationarization(config), self.spectral_gating = SpectralGating(config), self.adaptive_scanning = AdaptiveSelectiveScanning(config), self.mlp = nn.Sequential(nn.Linear(config.d_model, 4 * config.d_model), nn.SiLU(), nn.Linear(4 * config.d_model, config.d_model))`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: SAMbaConfig -> None` | <ul><li>`super().__init__()`</li><li>`self.config = config`</li><li>`self.norm = nn.LayerNorm(config.d_model)`</li><li>`self.temporal_basis = TemporalBasisParameterization(config)`</li><li>`self.wavelet_stationarization = MultiScaleStationarization(config)`</li><li>`self.spectral_gating = SpectralGating(config)`</li><li>`self.adaptive_scanning = AdaptiveSelectiveScanning(config)`</li><li>`self.mlp = nn.Sequential(nn.Linear(config.d_model, 4 * config.d_model), nn.SiLU(), nn.Linear(4 * config.d_model, config.d_model))`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`residual = x  # [B, L, D]`</li><li>`x_norm = self.norm(x)  # [B, L, D]`</li><li>`A_t = self.temporal_basis(x_norm)  # [B, L, N, M]`</li><li>`if self.config.use_wavelet_stationarization:`</li><li>`    x_norm = self.wavelet_stationarization(x_norm)  # [B, L, D]`</li><li>`if self.config.use_spectral_gating:`</li><li>`    x_norm = self.spectral_gating(x_norm)  # [B, L, D]`</li><li>`y = self.adaptive_scanning(x_norm, A_t)  # [B, L, D]`</li><li>`x = residual + y  # [B, L, D]`</li><li>`residual = x  # [B, L, D]`</li><li>`x_norm = self.norm(x)  # [B, L, D]`</li><li>`mlp_out = self.mlp(x_norm)  # [B, L, D]`</li><li>`x = residual + mlp_out  # [B, L, D]`</li><li>`return x  # Output of SAMba block`</li></ul> |



---
### 📄 `src\models\samba.py`
> *Complete SAMba model for time series forecasting*

**Imports**: 
`import torch, import torch.nn as nn, from .samba_block import SAMbaBlock`

#### Classes
**`class SAMba(nn.Module)`**
- *Attributes*: `self.config = config, self.embedding = nn.Linear(input_dim, config.d_model), self.blocks = nn.ModuleList([SAMbaBlock(config) for _ in range(config.n_layers)]), self.head = nn.Linear(config.d_model, output_dim)`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `config: SAMbaConfig, input_dim: int, output_dim: int -> None` | <ul><li>`super().__init__()`</li><li>`self.config = config`</li><li>`self.embedding = nn.Linear(input_dim, config.d_model)`</li><li>`self.blocks = nn.ModuleList([SAMbaBlock(config) for _ in range(config.n_layers)])`</li><li>`self.head = nn.Linear(config.d_model, output_dim)`</li></ul> |
| **forward** | `x: torch.Tensor -> torch.Tensor` | <ul><li>`B, L, D_in = x.shape  # [batch, seq_len, input_dim]`</li><li>`x = self.embedding(x)  # [B, L, d_model]`</li><li>`for block in self.blocks:`</li><li>`    x = block(x)  # [B, L, d_model]`</li><li>`output = self.head(x)  # [B, L, output_dim]`</li><li>`return output  # Forecasted sequence`</li></ul> |



---
### 📄 `src\data\time_series_loader.py`
> *Data loading and preprocessing for time series*

**Imports**: 
`import torch, from torch.utils.data import Dataset, DataLoader, import numpy as np`

#### Classes
**`class TimeSeriesDataset(Dataset)`**
- *Attributes*: `self.data = data, self.seq_len = seq_len, self.pred_len = pred_len`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `data: torch.Tensor, seq_len: int, pred_len: int -> None` | <ul><li>`self.data = data`</li><li>`self.seq_len = seq_len`</li><li>`self.pred_len = pred_len`</li></ul> |
| **__len__** | ` -> int` | <ul><li>`return len(self.data) - self.seq_len - self.pred_len + 1`</li></ul> |
| **__getitem__** | `idx: int -> Tuple[torch.Tensor, torch.Tensor]` | <ul><li>`x = self.data[idx:idx+self.seq_len]  # [seq_len, input_dim]`</li><li>`y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]  # [pred_len, output_dim]`</li><li>`return x, y`</li></ul> |



---
### 📄 `src\training\trainer.py`
> *Training loop and optimization for SAMba*

**Imports**: 
`import torch, import torch.nn as nn, from torch.optim import AdamW, from torch.utils.data import DataLoader`

#### Classes
**`class SAMbaTrainer()`**
- *Attributes*: `self.model = model, self.config = config, self.optimizer = AdamW(model.parameters(), lr=learning_rate), self.criterion = nn.MSELoss()`

| Method | Signature | Logic (Pseudo-code) |
| :--- | :--- | :--- |
| **__init__** | `model: nn.Module, config: SAMbaConfig, learning_rate: float = 1e-4 -> None` | <ul><li>`self.model = model`</li><li>`self.config = config`</li><li>`self.optimizer = AdamW(model.parameters(), lr=learning_rate)`</li><li>`self.criterion = nn.MSELoss()`</li></ul> |
| **train_epoch** | `train_loader: DataLoader -> float` | <ul><li>`self.model.train()`</li><li>`total_loss = 0.0`</li><li>`for batch_idx, (x, y) in enumerate(train_loader):`</li><li>`    self.optimizer.zero_grad()`</li><li>`    pred = self.model(x)  # [B, pred_len, output_dim]`</li><li>`    loss = self.criterion(pred, y)`</li><li>`    loss.backward()`</li><li>`    self.optimizer.step()`</li><li>`    total_loss += loss.item()`</li><li>`return total_loss / len(train_loader)`</li></ul> |
| **validate** | `val_loader: DataLoader -> float` | <ul><li>`self.model.eval()`</li><li>`total_loss = 0.0`</li><li>`with torch.no_grad():`</li><li>`    for x, y in val_loader:`</li><li>`        pred = self.model(x)  # [B, pred_len, output_dim]`</li><li>`        loss = self.criterion(pred, y)`</li><li>`        total_loss += loss.item()`</li><li>`return total_loss / len(val_loader)`</li></ul> |



---
### 📄 `main.py`
> *Main execution script for training and evaluation*

**Imports**: 
`import torch, from src.config import SAMbaConfig, from src.models.samba import SAMba, from src.data.time_series_loader import TimeSeriesDataset, from src.training.trainer import SAMbaTrainer, from torch.utils.data import DataLoader, random_split`


#### Functions
- 🔹 **`main`**
  - Logic:
1. `config = SAMbaConfig(d_model=512, d_state=128, n_layers=12, n_basis=8, n_experts=4)`
1. `model = SAMba(config, input_dim=1, output_dim=1)`
1. `dataset = TimeSeriesDataset(data, seq_len=96, pred_len=24)`
1. `train_size = int(0.8 * len(dataset))`
1. `val_size = len(dataset) - train_size`
1. `train_dataset, val_dataset = random_split(dataset, [train_size, val_size])`
1. `train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)`
1. `val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)`
1. `trainer = SAMbaTrainer(model, config, learning_rate=1e-4)`
1. `for epoch in range(100):`
1. `    train_loss = trainer.train_epoch(train_loader)`
1. `    val_loss = trainer.validate(val_loader)`
1. `    print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}')`

---

## 3. 主流程 (Execution Flow)
1. Load and preprocess time series data -> 2. Initialize SAMba model with config -> 3. Create data loaders -> 4. Train for multiple epochs -> 5. Validate performance -> 6. Save best model

---
<!-- SYSTEM SEPARATOR -->

# 🟢 用户决策区

**决策 (Action)**: [ APPROVE ] 

**反馈意见 (Feedback)**:
<!-- 比如：forward 函数里的 shape 好像不对，应该是 [B, D, L] -->