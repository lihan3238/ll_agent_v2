# 🏗️ 深度架构蓝图: 03_Architect_Round_1

## 1. 系统概览
- **Project**: `TimeMixer`
- **Data Flow**: 
```text
Input Data -> [Batch, Time, Features] -> Model (TimeMixer) -> Output Data [Batch, Time, Predictions]
2. 核心文件结构 (Detailed Specs)


📄 文件: src\data\data_preprocessing.py

Handles data loading and preprocessing.
Imports: import pandas as pd, import numpy as np



🔧 Functions


🔹 load_data

Logic: 1. Use pd.read_csv(file_path) to read the data. -> 2. Return the loaded DataFrame.

🔹 preprocess_data

Logic: 1. Normalize the data using Min-Max scaling. -> 2. Convert the DataFrame to a NumPy array. -> 3. Return the processed data.


📄 文件: src\models\time_mixer.py

Defines the TimeMixer model architecture.
Imports: import torch, import torch.nn as nn


📦 Classes


Class TimeMixer (inherits nn.Module)

Attrs: num_scales: int, input_dim: int, pdm_layers: nn.ModuleList, fmm_layer: nn.Linear

Methods:

🔹 __init__(self, num_scales: int, input_dim: int) -> None

📝 Initializes the TimeMixer model with specified scales and input dimension.

⚙️ Logic:

1. Call super().__init__() to initialize the parent class.

2. Set self.num_scales to num_scales.

3. Set self.input_dim to input_dim.

4. Initialize self.pdm_layers as nn.ModuleList of PDM layers.

5. Initialize self.fmm_layer as nn.Linear with input dimension and output dimension.

🔹 forward(self, x: torch.Tensor) -> torch.Tensor

📝 Computes the forward pass of the TimeMixer model.

⚙️ Logic:

1. Initialize an empty list to store outputs from PDM layers.

2. For each layer in self.pdm_layers:

   a. Apply layer to the input x.

   b. Append the output to the outputs list.

3. Concatenate outputs from all PDM layers along the last dimension.

4. Pass the concatenated outputs to self.fmm_layer to obtain final predictions.

5. Return the final predictions.



📄 文件: src\training\train.py

Contains the training loop for the TimeMixer model.
Imports: import torch, import torch.optim as optim, from src.models.time_mixer import TimeMixer, from src.data.data_preprocessing import load_data, preprocess_data



🔧 Functions


🔹 train_model

Logic: 1. Set the model to training mode: model.train(). -> 2. Create an optimizer using optim.Adam with model parameters and learning_rate. -> 3. For each epoch in range(epochs): ->    a. For each batch in train_loader: ->       i. Zero the gradients: optimizer.zero_grad(). ->       ii. Forward pass: get predictions from model using batch data. ->       iii. Calculate loss using a loss function (e.g., MSELoss). ->       iv. Backward pass: loss.backward(). ->       v. Update model parameters: optimizer.step().


3. 配置与依赖

Hyperparams: {'num_scales': '3', 'input_dim': '10', 'epochs': '50', 'learning_rate': '0.001'}

Requirements: torch==1.10.0, numpy==1.21.0

<!-- SYSTEM SEPARATOR -->

🟢 用户决策区

决策 (Action): [ APPROVE ]

反馈意见 (Feedback):

<!-- 比如：forward 函数里的 shape 好像不对，应该是 [B, D, L] -->