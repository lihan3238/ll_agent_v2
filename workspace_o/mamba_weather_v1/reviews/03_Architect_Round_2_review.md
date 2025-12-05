# 🏗️ 深度架构蓝图: 03_Architect_Round_2

## 1. 系统概览
- **Project**: `TimeMixer`
- **Data Flow**: 
```text
Data -> [Batch, Time, Features] -> Model -> [Batch, Time, Features]
2. 核心文件结构 (Detailed Specs)


📄 文件: src\data\preprocessing.py

Data preprocessing functions for time series data.
Imports: import numpy as np, import pandas as pd



🔧 Functions


🔹 prepare_data

Logic: Extract target variable from data using target_column. -> Normalize the data to have zero mean and unit variance. -> Convert the processed data into a PyTorch tensor. -> Return the tensor with shape [Batch, Time, Features].


📄 文件: src\models\time_mixer.py

Core model class for the TimeMixer architecture.
Imports: import torch, import torch.nn as nn


📦 Classes


Class TimeMixer (inherits nn.Module)

Attrs: num_scales: int, num_features: int, pdm_layers: nn.ModuleList, fmm_layer: nn.Linear

Methods:

🔹 __init__(self, num_scales: int, num_features: int) -> None

📝 Initializes the TimeMixer model with the specified number of scales and features.

⚙️ Logic:

Set self.num_scales to num_scales.

Set self.num_features to num_features.

Initialize self.pdm_layers as an empty nn.ModuleList.

For each scale from 0 to num_scales - 1:

  Append a new PDM layer to self.pdm_layers.

Initialize self.fmm_layer as a linear layer with input size num_scales * num_features and output size num_features.

🔹 forward(self, x: torch.Tensor) -> torch.Tensor

📝 Forward pass through the TimeMixer model.

⚙️ Logic:

Ensure x has shape [Batch, Time, Features].

Initialize an empty list predictions.

For each pdm_layer in self.pdm_layers:

  Apply pdm_layer to x and append the output to predictions.

Concatenate predictions along the last dimension to form a tensor of shape [Batch, Time, num_scales * Features].

Apply self.fmm_layer to the concatenated predictions.

Return the output tensor with shape [Batch, Time, Features].



📄 文件: src\training\train.py

Training loop for the TimeMixer model.
Imports: import torch, import torch.optim as optim, from src.models.time_mixer import TimeMixer, from src.data.preprocessing import prepare_data



🔧 Functions


🔹 train_model

Logic: Set optimizer to optim.Adam with model parameters and learning_rate. -> For epoch in range(num_epochs): ->   For batch in train_loader: ->     Get input data and target from batch. ->     Zero the optimizer gradients. ->     Perform forward pass: output = model(input). ->     Compute loss using a suitable loss function. ->     Backpropagate the loss. ->     Update model parameters using optimizer.


3. 配置与依赖

Hyperparams: {'num_scales': '3', 'num_features': '5', 'num_epochs': '100', 'learning_rate': '0.001'}

Requirements: torch==1.13.0, numpy==1.23.0

<!-- SYSTEM SEPARATOR -->

🟢 用户决策区

决策 (Action): [ APPROVE ]

反馈意见 (Feedback):

<!-- 比如：forward 函数里的 shape 好像不对，应该是 [B, D, L] -->