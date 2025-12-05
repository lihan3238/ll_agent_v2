# 🧠 理论框架审查: 02_Theory_Round_1

## 1. 研究领域 (Field)
`Time Series Forecasting with Non-Stationary State Space Models`

## 2. 问题定义 (Problem Formulation)
> 这里是问题的形式化定义（支持 LaTeX）：

Given an input multivariate time series $\mathbf{X} \in \mathbb{R}^{L \times V}$ where $L$ is sequence length and $V$ is number of variates, the goal is to forecast future values $\mathbf{Y} \in \mathbb{R}^{T \times V}$ under the constraint that real-world time series exhibit non-stationary statistical properties where temporal patterns evolve over time.

## 3. 核心方法 (Proposed Methodology)
We propose **Spectral Adaptive Mamba (SAMba)**, a novel framework that overcomes the linear time-invariant (LTI) constraint through time-varying parameterization of state transition matrices. The core architecture consists of three key components:

1. **Temporal Basis Parameterization**: The state transition matrix $\mathbf{A}$ becomes time-dependent via learnable basis functions: $\mathbf{A}(t) = \sum_{k=1}^{K} \alpha_k(t) \mathbf{B}_k$, where $\{\mathbf{B}_k\}_{k=1}^K$ are orthogonal temporal basis matrices and $\alpha_k(t)$ are input-dependent coefficients computed as $\alpha_k(t) = \text{softmax}(\mathbf{W}_k^\top \mathbf{x}_t)$.

2. **Multi-Scale Stationarization Module**: Employ wavelet decomposition to model time-varying statistical properties: $\mathbf{x}_t = \sum_{s} \mathbf{w}_{s,t} + \epsilon_t$, where $\mathbf{w}_{s,t}$ represents multi-scale temporal patterns.

3. **Differentiable Spectral Gating**: Dynamic routing of temporal patterns to specialized state space experts using attention-weighted spectral embeddings: $\mathbf{g}_t = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d})$.

4. **Adaptive Selective Scanning**: Modified scanning mechanism that maintains $O(L \log L)$ complexity through input-dependent time steps: $\Delta_t = \text{softplus}(\mathbf{W}_\Delta \mathbf{x}_t)$.

The complete forward pass is:

$\mathbf{h}_t = \mathbf{A}(t) \mathbf{h}_{t-1} + \mathbf{B}(t) \mathbf{x}_t$

$\mathbf{y}_t = \mathbf{C} \mathbf{h}_t + \mathbf{D} \mathbf{x}_t$

with time-varying parameters enabling adaptation to non-stationary temporal patterns.

## 4. 理论分析 (Analysis & Proofs)
**Theorem 1 (Expressivity)**: SAMba constitutes a strictly larger function class than LTI state space models.

*Proof Sketch*: The time-varying parameterization $\mathbf{A}(t)$ enables modeling of systems with time-varying dynamics, while standard Mamba models are restricted to LTI systems.

**Theorem 2 (Complexity)**: SAMba maintains $O(L \log L)$ computational complexity through adaptive selective scanning.

*Proof Sketch*: The discretization step size $\Delta_t$ becomes input-dependent, preserving the efficiency of selective state space models while gaining adaptivity to non-stationary patterns.

**Complexity Analysis**: The wavelet-based stationarization adds $O(L)$ complexity, while the spectral gating mechanism contributes $O(L \cdot E)$ where $E$ is number of experts, maintaining overall $O(L \log L)$ scaling.

## 5. 关键创新点 (Innovations)
- Time-varying state transition matrices via learnable temporal basis functions, breaking the LTI constraint
- Explicit multi-scale stationarization through wavelet decomposition for modeling time-varying statistical properties
- Differentiable spectral gating for dynamic expert routing based on temporal pattern characteristics
- Adaptive selective scanning mechanism that maintains computational efficiency while enabling non-stationary adaptation
- Theoretical guarantees on expanded function class while preserving sub-quadratic complexity

---
<!-- SYSTEM SEPARATOR -->

# 🟢 用户决策区

**决策 (Action)**: [ APPROVE ] 

**反馈意见 (Feedback)**:
<!-- 比如：公式推导有点问题，或者请增加对隐私安全性的讨论 -->