# YOLOPX-V2 平台架构与文件流程

**文档版本:** 1.0
**日期:** 2025-07-30

## **1. 研究目的与设计哲学**

本平台 (`YOLOPX/v2`) 的核心设计目标是服务于 `research_plan_v3.md` 中定义的宏观研究计划。它旨在成为一个**清晰、灵活、可扩展**的多任务学习实验平台，专门用于研究自动驾驶感知模型中的**任务冲突**问题。

**核心设计哲学:**

1.  **配置驱动一切:** 所有的实验变量，包括模型架构、数据集、训练超参数，都必须由 `.yaml` 配置文件定义。代码本身应保持最大限度的通用性，不包含任何硬编码的实验参数。
2.  **模块化与隔离:** 不同的模型架构（YOLOPv1, v3, X）应作为独立的、可插拔的模块存在，它们之间的代码应严格隔离，以保证官方实现的忠实性，避免交叉污染。
3.  **清晰的数据流:** 从启动训练到完成一次迭代，整个代码的调用链和数据流应清晰、明确，易于理解和调试。
4.  **为扩展而设计:** 平台的设计必须为未来集成新的模型、新的任务头、以及新的冲突优化算法（如 `阶段 2` 和 `阶段 3` 所规划的）预留清晰的接口。

## **2. 重构后的文件结构总览**

```
YOLOPX/v2/
├── train.py                # 唯一的训练主函数入口
├── refactoring.md          # 本次重构的日志
├── files_flow.md           # (本文件) 平台架构与流程说明
│
├── cfgs/                   # 存放所有 .yaml 配置文件
│   ├── data/
│   │   └── bdd100k.yaml
│   ├── models/
│   │   ├── yolopx_v2_anchor_free.yaml
│   │   ├── yolop_v1_official.yaml
│   │   └── yolop_v3_official.yaml
│   └── train_v2.yaml       # 统一的训练超参数配置
│
├── core/                   # 存放核心算法逻辑
│   └── loss.py             # 损失函数的定义与管理
│
├── data/                   # 存放数据集定义与加载逻辑
│   └── autodrive_dataset.py
│
├── models/                 # 存放所有模型架构与模块
│   ├── builder.py            # 动态模型构建器
│   ├── common_modules.py     # YOLOPX (v2) 的共享基础模块
│   ├── heads/                # 存放不同的检测头实现
│   │   ├── yolop_head.py     # Anchor-based (YOLOPv1-style)
│   │   └── yolox_head.py     # Anchor-free (YOLOPX-style)
│   ├── yolop_v1_official_modules/ # YOLOPv1 官方模块的隔离包
│   │   └── modules.py
│   └── yolop_v3_official_modules/ # YOLOPv3 官方模块的隔离包
│       └── modules.py
│
└── utils/                  # 存放通用的工具函数
    ├── autoanchor.py
    ├── dataloader.py
    └── utils.py
```

## **3. 核心执行流程**

下图描述了从用户执行命令到完成一次训练迭代的完整流程：

```mermaid
graph TD
    A[用户执行: python train.py --model_cfg A --data_cfg B --train_cfg C] --> B(train.py: main());
    B --> C{加载 YAML 配置};
    C --> D[合并成统一的 cfg 对象];
    D --> E[初始化 Dataset & DataLoader];
    D --> F[调用 models.builder.get_net_from_yaml];
    F --> G{builder.py: 检查 cfg 中的 'model_family'};
    G --> H1[加载 yolop_v1_official_modules];
    G --> H2[加载 yolop_v3_official_modules];
    G --> H3[加载 v2/common_modules (默认)];
    H1 --> I[构建 YOLOPv1 模型];
    H2 --> I[构建 YOLOPv3 模型];
    H3 --> I[构建 YOLOPX 模型];
    I --> J[返回 Model 实例];
    D --> K[初始化 Optimizer];
    D --> L[调用 core.loss.get_loss];
    L --> M[返回 Criterion 实例];
    
    subgraph "训练循环 (Training Loop)"
        direction LR
        N[For-loop: for epoch in epochs];
        N --> O[For-loop: for data in dataloader];
        O --> P[模型前向传播: outputs = model(data)];
        P --> Q[计算损失: loss = criterion(outputs, targets)];
        Q --> R[反向传播: loss.backward()];
        R --> S[优化器更新: optimizer.step()];
    end

    E --> O;
    J --> P;
    M --> Q;
    K --> S;
```

## **4. 冲突检测算法集成点**

根据 `research_plan_v3.md` 的**阶段 2**，我们需要集成 `PCGrad`, `CAGrad`, `TAG` 等冲突优化算法。这些算法将在训练循环的梯度计算和更新步骤之间被调用。

```mermaid
graph TD
    subgraph "训练循环 (Training Loop)"
        direction LR
        ... --> Q[计算损失: loss = criterion(outputs, targets)];
        Q --> R1(**冲突优化算法介入点**);
        R1 --> R2{IF method == 'PCGrad'};
        R2 --> R3[计算各任务梯度, 执行梯度手术];
        R3 --> S[反向传播 (使用修改后的梯度)];
        R1 --> T1{IF method == 'GradNorm'};
        T1 --> T2[计算各任务梯度范数, 更新损失权重];
        T2 --> Q;
        R1 --> U1{IF method == 'TAG'};
        U1 --> P[模型内部通过注意力机制自动完成];
        S --> ...
    end
```
**实现策略:** 我们将在 `train.py` 的训练循环中，`loss.backward()` 之前，插入一个 `solver` 模块的调用。这个 `solver` 将根据配置 (`cfg.TRAIN.SOLVER`) 来决定执行哪种冲突优化算法。

## **5. 未来扩展性接口 (Future Extension)**

本平台为 `research_plan_v3.md` 的后续阶段预留了清晰的扩展接口：

*   **阶段 3 (创新方法):**
    *   **新模型:** 只需在 `models/` 下创建新的模块/架构文件，并在 `cfgs/models/` 下创建新的 `.yaml` 配置文件。`builder.py` 的动态加载机制将自动识别。
    *   **新优化器:** 只需在 `core/` 下创建一个新的 `solvers/` 目录，实现新的优化器算法，然后在 `train.py` 的 `solver` 调用逻辑中增加一个分支即可。

*   **阶段 4 (任务扩展):**
    *   **新任务头:** 只需在 `models/heads/` 下创建新的头文件（例如 `depth_head.py`），在 `builder.py` 中注册，并在 `.yaml` 中定义其连接方式。
    *   **新数据集:** 只需在 `data/` 下创建新的数据集定义文件，并在 `cfgs/data/` 下创建新的 `.yaml` 配置文件。
