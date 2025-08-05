# Unified YOLOP Training Framework

这是一个统一的YOLOP系列模型训练框架，支持在单一代码库中训练和评估YOLOP v1、YOLOPv2、YOLOP v3和YOLOPX四种模型。

## 项目特点

- **统一接口**: 使用相同的训练、验证和推理脚本运行所有模型
- **配置驱动**: 通过配置文件选择模型类型和训练参数
- **任务冲突检测**: 集成了多任务学习的冲突检测和分析功能
- **参数文件化**: 所有参数都可通过配置文件管理

## 目录结构

```
unified_yolop/
├── configs/          # 配置文件
│   ├── default.yaml  # 默认配置
│   └── models/       # 各模型专用配置
├── core/            # 核心功能模块
│   ├── loss.py      # 损失函数
│   └── metrics.py   # 评估指标
├── data/            # 数据处理
│   └── dataset.py   # 数据集类
├── models/          # 模型定义
│   └── factory.py   # 模型工厂
├── utils/           # 工具函数
│   ├── config.py    # 配置管理
│   └── conflict.py  # 冲突检测
├── train.py         # 训练脚本
├── validate.py      # 验证脚本
├── inference.py     # 推理脚本
└── logs/            # 训练日志
```

## 使用方法

### 训练模型

```bash
# 训练YOLOPX (anchor-free)
python train.py --model yolopx

# 训练YOLOP v1 (anchor-based)
python train.py --model yolop_v1

# 训练YOLOP v3 (遥感优化)
python train.py --model yolop_v3

# 使用自定义配置
python train.py --model yolopx --config configs/custom.yaml
```

### 数据准备

数据集路径: `/workspace/bdd100k/yolop_train/`

确保数据集包含以下结构：
- images/100k/{train,val,test}/
- labels/det/{train,val,test}/
- labels/da_seg/{train,val,test}/
- labels/ll_seg/{train,val,test}/

## 任务冲突检测

框架会自动检测并记录任务间的冲突：
- 梯度冲突分析
- 损失比率分析
- 冲突历史记录

## 开发日志

所有开发进度记录在 `log.md` 文件中。