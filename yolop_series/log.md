# YOLOP Series 统一框架开发日志

## 2025-07-31 项目初始化

### 任务概述
创建一个统一的框架，能够在单一代码库中运行YOLOP v1, YOLOPv2, YOLOP v3和YOLOPX四种模型的训练、验证和推理代码。
- 数据集路径：/workspace/bdd100k/yolop_train/
- 主要目标：集成任务冲突检测和解决方法

### 完成的任务

#### 1. 项目目录结构创建
创建了基础目录结构：
```
yolop_series/
├── models/       # 存放各版本模型定义
├── utils/        # 工具函数和通用模块
├── configs/      # 配置文件
├── data/         # 数据加载相关代码
├── logs/         # 训练日志
└── scripts/      # 辅助脚本
```

#### 2. 初步分析
- 阅读了research_plan.md，了解到：
  - 初步发现：Anchor-based检测头比Anchor-free检测头表现出更低（约25%）的任务冲突
  - 需要系统性地对比YOLOP家族模型的冲突特性
  - 最终目标是提出创新的冲突解决方案

### 下一步计划
1. Git clone YOLOPX源代码 ✅ (已存在于/workspace/YOLOPX)
2. 分析四个模型的代码结构
3. 设计统一的代码架构

## 2025-07-31 代码结构分析

### YOLOPX代码结构
成功克隆YOLOPX代码，主要结构：
- `lib/models/`: 模型定义（YOLOP.py, YOLOX_Head_scales_noshare.py）
- `lib/core/`: 核心功能（loss.py, function.py, evaluate.py）
- `lib/dataset/`: 数据集处理（bdd.py, AutoDriveDataset.py）
- `lib/config/`: 配置系统（default.py）
- `tools/`: 训练、测试、演示脚本
- 特点：采用YOLOX风格的anchor-free检测头

### YOLOP v1代码结构
- 与YOLOPX结构相似，但使用anchor-based检测头
- 配置使用yacs库的CfgNode
- 模型定义在lib/models/YOLOP.py
- 使用CSP-Darknet作为backbone

### YOLOPv2代码结构  
- 精简版结构，只有demo.py和utils
- 使用torchscript加载模型（.pt文件）
- 没有训练代码，仅推理功能

### YOLOP v3代码结构
- 代码结构与v1相似
- 使用ELANNet作为backbone（YOLOv7风格）
- 特点：PaFPNELAN结构，SimAM注意力机制
- 针对遥感场景优化，使用anchor-based检测

## 2025-07-31 统一框架设计

### 已完成工作
1. 创建了统一的配置系统
   - `configs/default.yaml`: 默认配置
   - `configs/models/`: 各模型特定配置
   - 支持配置继承和覆盖

2. 实现了配置管理器
   - `utils/config.py`: 配置加载和管理
   - 支持YAML文件加载和命令行覆盖

3. 创建了模型工厂
   - `models/model_factory.py`: 统一的模型创建接口
   - 根据配置自动选择和创建对应模型

### 统一架构设计
- 单一入口：train.py, validation.py, inference.py
- 配置驱动：通过配置文件选择模型类型
- 模块化设计：模型、数据集、损失函数等独立模块
- 任务冲突检测：集成到训练循环中

## 2025-07-31 核心模块实现

### 已实现模块
1. **数据加载模块** (`data/dataset.py`)
   - 统一的YOLOPDataset类
   - 支持BDD100K数据集格式
   - 处理检测标签和分割掩码
   - 自定义collate函数

2. **任务冲突检测** (`utils/conflict_detector.py`)
   - 梯度冲突检测
   - 损失比率冲突检测
   - 冲突历史记录和统计

3. **训练脚本** (`train_unified.py`)
   - 统一的训练入口
   - 支持所有模型变体
   - 集成冲突检测
   - 检查点保存和恢复

### 待完成工作
- 完整的模型集成（需要解决依赖问题）
- 验证和推理脚本
- YOLOPv2的训练支持
- 更详细的损失函数实现