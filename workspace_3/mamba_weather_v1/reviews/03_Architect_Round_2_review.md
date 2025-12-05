# 🏗️ 深度架构蓝图: 03_Architect_Round_2

## 1. 系统概览
- **Project**: `Stormer`
- **Data Flow**: 
```text
Data -> [Batch, Time, Features] -> Model -> [Batch, Time, Output]
2. 核心文件结构 (Detailed Specs)


📄 文件: src\data\preprocessing.py

Data preprocessing utilities for loading and transforming weather data.
Imports: import pandas as pd, import numpy as np



🔧 Functions


🔹 load_data

Logic: Read the CSV file into a DataFrame. -> Return the DataFrame.

🔹 normalize_data

Logic: Calculate the mean and standard deviation of the data. -> Subtract the mean and divide by the standard deviation for each column. -> Return the normalized DataFrame.


📄 文件: src\models\stormer.py

Implementation of the Stormer model based on transformer architecture.
Imports: import torch, import torch.nn as nn


📦 Classes


Class StormerModel (inherits nn.Module)

Attrs: embedding: nn.Linear, transformer_blocks: nn.ModuleList, layer_norm: nn.LayerNorm, output_layer: nn.Linear

Methods:

🔹 __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None

📝 Initialize the Stormer model with embedding and transformer blocks.

⚙️ Logic:

Initialize embedding layer with input_dim.

Create an empty ModuleList for transformer_blocks.

For i in range(num_layers):

    Append a new transformer block to transformer_blocks.

Initialize layer normalization.

Initialize output layer with hidden_dim.

🔹 forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor

📝 Forward pass through the Stormer model.

⚙️ Logic:

Input x has shape [Batch, Time, Features].

Pass x through the embedding layer to get embedded_x.

For each transformer block in transformer_blocks:

    Update embedded_x by passing it through the block.

Apply layer normalization to the final output.

Pass the normalized output through the output layer.

Return the final output.



📄 文件: src\train\training_loop.py

Training loop for the Stormer model.
Imports: import torch, import torch.optim as optim, from src.models.stormer import StormerModel, from src.data.preprocessing import load_data, normalize_data



🔧 Functions


🔹 train_model

Logic: Initialize the optimizer with model parameters and learning_rate. -> For epoch in range(num_epochs): ->     For batch in train_loader: ->         Get input data and labels from batch. ->         Zero the gradients. ->         Forward pass through the model to get predictions. ->         Calculate the loss using the pressure-weighted loss function. ->         Backpropagate the loss. ->         Update model parameters using optimizer.


3. 配置与依赖

Hyperparams: {'learning_rate': '0.001', 'num_epochs': '100', 'batch_size': '32', 'input_dim': '10', 'hidden_dim': '64', 'num_layers': '6'}

Requirements: torch==1.12.1, numpy==1.21.2, pandas==1.3.3, scikit-learn==0.24.2, matplotlib==3.4.3

<!-- SYSTEM SEPARATOR -->

🟢 用户决策区

决策 (Action): [ APPROVE ]

反馈意见 (Feedback):

<!-- 比如：forward 函数里的 shape 好像不对，应该是 [B, D, L] -->