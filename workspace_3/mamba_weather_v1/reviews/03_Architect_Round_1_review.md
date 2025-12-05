# 🏗️ 深度架构蓝图: 03_Architect_Round_1

## 1. 系统概览
- **Project**: `Stormer`
- **Data Flow**: 
```text
Input Data -> [Batch, Sequence Length, Variables] -> Model -> [Batch, Output Variables]
2. 核心文件结构 (Detailed Specs)


📄 文件: src\data\preprocessing.py

Data preprocessing routines for input time series data.
Imports: import pandas as pd, import numpy as np



🔧 Functions


🔹 prepare_data

Logic: 1. Initialize an empty list for sequences. -> 2. For each index from 0 to len(data) - sequence_length: ->    a. Append data[index:index + sequence_length] to sequences. -> 3. Convert sequences to a NumPy array and return.


📄 文件: src\models\stormer.py

Model definition for the Stormer transformer architecture.
Imports: import torch, import torch.nn as nn


📦 Classes


Class StormerModel (inherits nn.Module)

Attrs: embedding_layer: nn.Embedding, transformer_blocks: nn.ModuleList, output_layer: nn.Linear

Methods:

🔹 __init__(self, num_variables: int, hidden_dim: int, num_layers: int) -> None

📝 Initializes the Stormer model components.

⚙️ Logic:

1. Call the parent class constructor.

2. Initialize embedding_layer with nn.Embedding(num_variables, hidden_dim).

3. Initialize transformer_blocks as an empty ModuleList.

4. For i in range(num_layers):

   a. Append a new transformer block to transformer_blocks.

5. Initialize output_layer with nn.Linear(hidden_dim, num_variables).

🔹 forward(self, x: torch.Tensor) -> torch.Tensor

📝 Performs the forward pass through the model.

⚙️ Logic:

1. Pass x through the embedding_layer to get embeddings.

2. For each transformer block in transformer_blocks:

   a. Apply the block to the embeddings.

3. Pass the final output through the output_layer.

4. Return the final output.



📄 文件: src\training\train.py

Training loop for the Stormer model.
Imports: import torch, import torch.optim as optim, from src.models.stormer import StormerModel, from src.data.preprocessing import prepare_data



🔧 Functions


🔹 train_model

Logic: 1. Set model to training mode. -> 2. Initialize optimizer as optim.Adam with model parameters and learning_rate. -> 3. For epoch in range(epochs): ->    a. For each batch in train_data: ->       i. Zero the gradients. ->       ii. Get model predictions by calling model.forward on the batch. ->       iii. Calculate loss using pressure-weighted loss function. ->       iv. Backpropagate the loss. ->       v. Update model parameters using optimizer.


3. 配置与依赖

Hyperparams: {'num_layers': '6', 'hidden_dim': '128', 'num_variables': '10', 'sequence_length': '30', 'learning_rate': '0.001', 'epochs': '50'}

Requirements: torch==1.12.0, numpy==1.21.0, pandas==1.3.0

<!-- SYSTEM SEPARATOR -->

🟢 用户决策区

决策 (Action): [ APPROVE ]

反馈意见 (Feedback):

<!-- 比如：forward 函数里的 shape 好像不对，应该是 [B, D, L] -->