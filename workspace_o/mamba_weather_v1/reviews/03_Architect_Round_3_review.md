# 🏗️ 深度架构蓝图: 03_Architect_Round_3

## 1. 系统概览
- **Project**: `TimeMixer`
- **Data Flow**: 
```text
Data -> [Batch, Time, Features] -> [PDM Block] -> [Seasonal, Trend] -> [FMM Block] -> Output [Batch, Time, Predictions]
2. 核心文件结构 (Detailed Specs)


📄 文件: src\data\preprocessing.py

Data preprocessing for time series.
Imports: numpy as np, pandas as pd



🔧 Functions


🔹 decompose_time_series

Logic: Apply seasonal decomposition on the data. -> Extract seasonal component. -> Extract trend component. -> Return seasonal and trend components as numpy arrays.


📄 文件: src\models\time_mixer.py

Implementation of the TimeMixer architecture.
Imports: torch, torch.nn as nn


📦 Classes


Class TimeMixer (inherits nn.Module)

Attrs: num_scales: int, num_layers: int, seasonal_layer: nn.Module, trend_layer: nn.Module, predictors: List[nn.Module]

Methods:

🔹 __init__(self, num_scales: int, num_layers: int) -> None

📝 Initialize the TimeMixer model with given scales and layers.

⚙️ Logic:

Set self.num_scales to num_scales.

Set self.num_layers to num_layers.

Initialize seasonal_layer as an instance of nn.Module.

Initialize trend_layer as an instance of nn.Module.

Initialize predictors as an empty list.

For each scale in range(num_scales):

    Append a new predictor module to self.predictors.

🔹 forward(self, x: torch.Tensor) -> torch.Tensor

📝 Perform forward pass through the TimeMixer model.

⚙️ Logic:

Input x has shape [Batch, Time, Features].

Decompose x into seasonal and trend components using seasonal_layer and trend_layer.

For each scale in range(self.num_scales):

    Compute predictions from each predictor on seasonal and trend components.

Aggregate the predictions from all predictors.

Return the aggregated predictions.



📄 文件: src\training\train.py

Training loop for the TimeMixer model.
Imports: torch, torch.optim as optim, torch.nn.functional as F, src.models.time_mixer.TimeMixer



🔧 Functions


🔹 train_model

Logic: Set optimizer as optim.Adam for the model parameters with learning rate. -> For epoch in range(num_epochs): ->     For batch in train_loader: ->         Get input data x and target y from batch. ->         Zero the gradients. ->         Compute model output by passing x through model. ->         Compute loss using F.mse_loss between output and target. ->         Backpropagate the loss. ->         Update model parameters using optimizer.


3. 配置与依赖

Hyperparams: {'num_scales': '3', 'num_layers': '2', 'learning_rate': '0.001', 'num_epochs': '50'}

Requirements: torch==1.12.0, numpy==1.21.0, pandas==1.3.0

<!-- SYSTEM SEPARATOR -->

🟢 用户决策区

决策 (Action): [ APPROVE ]

反馈意见 (Feedback):

<!-- 比如：forward 函数里的 shape 好像不对，应该是 [B, D, L] -->