# Unified YOLOP 项目总结

## 项目概述

我已经在 `/workspace/YOLOP_conflict/unified_yolop/` 目录下创建了一个完整的统一训练框架，支持YOLOP系列的4个模型变体。

## 核心特性

### 1. 统一的训练入口
- 单一的 `train.py` 文件可以训练所有模型
- 通过 `--model` 参数选择模型类型：`yolop_v1`, `yolop_v3`, `yolopx`
- YOLOPv2 需要预训练模型支持

### 2. 配置驱动设计
- 默认配置：`configs/default.yaml`
- 模型特定配置：`configs/models/` 目录
- 支持命令行参数覆盖

### 3. 任务冲突检测
- 梯度冲突分析
- 损失比率分析
- 支持多种冲突解决方法（PCGrad, CAGrad）
- 实时记录和分析冲突指标

### 4. 模块化架构
```
unified_yolop/
├── configs/        # 配置文件
├── core/          # 核心功能（损失函数、评估指标）
├── data/          # 数据加载
├── models/        # 模型工厂
├── utils/         # 工具函数（配置、冲突检测）
└── train.py       # 主训练脚本
```

## 使用方法

### 基础训练命令
```bash
# 训练YOLOPX（anchor-free）
python train.py --model yolopx

# 训练YOLOP v1（anchor-based）
python train.py --model yolop_v1

# 训练YOLOP v3（遥感优化）
python train.py --model yolop_v3
```

### 高级选项
```bash
# 指定训练参数
python train.py --model yolopx --epochs 100 --batch-size 16 --device cuda:0

# 使用自定义配置
python train.py --model yolop_v1 --config my_config.yaml

# 恢复训练
python train.py --model yolopx --resume checkpoint.pth
```

### 使用示例脚本
```bash
./run_example.sh yolopx --epochs 10
```

## 技术实现

### 模型工厂（ModelFactory）
- 根据配置创建不同的模型实例
- ModelWrapper提供统一的输出接口
- 支持加载原始代码库的模型

### 任务冲突检测（TaskConflictDetector）
- 计算任务间的梯度余弦相似度
- 监控损失比率的不平衡
- 提供冲突解决策略

### 数据加载（UnifiedYOLOPDataset）
- 支持BDD100K数据格式
- 同时处理检测标签和分割掩码
- 自适应图像缩放和填充

## 注意事项

1. **模型集成**：完整的模型集成需要正确设置原始代码库的依赖
2. **YOLOPv2**：只支持加载预训练模型，需要提供.pt文件路径
3. **数据路径**：确保BDD100K数据集位于 `/workspace/bdd100k/yolop_train/`
4. **损失函数**：当前实现是简化版，建议根据需要集成原始损失函数

## 研究意义

这个框架为研究任务冲突提供了良好的平台：
- 可以直接对比anchor-based和anchor-free的冲突差异
- 支持实时监控和记录冲突指标
- 易于添加新的冲突解决方法
- 为博士论文研究提供实验基础

## 新增功能（已完成）

### 1. Wandb集成
- 自动记录所有训练指标和任务冲突数据
- 支持实时监控和模型对比
- 配置文件中设置 `LOG.WANDB: true` 即可启用

### 2. 任务冲突结果保存
- 每个模型训练结束后自动保存 `conflict_results.json`
- 包含完整的冲突统计：梯度冲突、损失比率等
- 方便进行模型间的冲突对比分析

### 3. 验证图片可视化
- 验证时自动保存预测结果图片
- 绿色显示可行驶区域，红色显示车道线
- 保存在 `runs/<exp>/val_epoch_<N>/` 目录

### 4. 测试和演示脚本
- `test_models.py`: 测试所有模型是否能正常加载
- `demo_train.py`: 快速演示训练流程

## 运行示例

```bash
# 测试模型加载
python3 test_models.py

# 运行训练演示（每个模型2个epoch）
python3 demo_train.py

# 正式训练
python3 train.py --model yolopx --epochs 100
python3 train.py --model yolop_v1 --epochs 100
python3 train.py --model yolop_v3 --epochs 100
```

## 查看结果

1. **TensorBoard**: 
   ```bash
   tensorboard --logdir runs/
   ```

2. **Wandb**: 
   - 访问 https://wandb.ai/
   - 查看 "unified-yolop" 项目

3. **冲突分析结果**:
   - `runs/<exp>/conflict_results.json`
   - 包含每个模型的冲突统计数据

## 后续工作

1. 完善原始模型的集成（需要设置依赖）
2. 实现独立的 `validate.py` 和 `inference.py`
3. 添加更多冲突解决算法（TAG等）
4. 支持YOLOPv2的训练（目前只支持推理）
5. 添加多类别检测支持