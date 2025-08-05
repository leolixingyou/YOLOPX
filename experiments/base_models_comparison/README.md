# YOLOP系列模型任务冲突检测实验

本目录包含了YOLOP系列模型（YOLOPv1, v2, v3, YOLOPX）的任务冲突检测实验代码和结果。

## 实验目的

评估多任务学习中不同任务（目标检测、驾驶区域分割、车道线分割）之间的梯度冲突程度，通过计算任务冲突强度（Task Conflict Intensity, TCI）来量化冲突。

## 主要文件

### 核心实验代码
- `simple_conflict_detection.py` - 简化的冲突检测实验实现
- `model_adapter.py` - 统一的模型接口适配器
- `run_conflict_detection.py` - 原始冲突检测运行脚本
- `fixed_train.py` - 修正的训练脚本

### 实验结果
- `experiment_report.md` - 详细的实验报告
- `visualize_tci_results.py` - TCI结果可视化脚本
- `results/` - 包含可视化图表和JSON格式的实验数据
  - `tci_comparison.png` - TCI对比图
  - `tci_distribution.png` - TCI分布图
- `runs/conflict_detection/` - 各模型的详细结果

### 配置文件
- `configs/bdd100k_experiment.yaml` - BDD100K数据集实验配置

## 实验结果摘要

| 模型 | 最终TCI | 平均TCI | 标准差 |
|------|---------|---------|--------|
| YOLOPv1 | 0.2151 | 0.2151 | 0.0170 |
| YOLOPv2 | 0.2129 | 0.2129 | 0.0227 |
| YOLOPv3 | 0.2241 | 0.2241 | 0.0244 |

## 使用方法

运行简化的冲突检测实验：
```bash
python3 simple_conflict_detection.py --epochs 5
```

生成可视化图表：
```bash
python3 visualize_tci_results.py
```

## 注意事项

1. 由于原始YOLOP模型存在架构兼容性问题，实验使用了简化的模型架构
2. TCI值在0.19-0.26之间表明任务间存在明显的梯度冲突
3. 建议使用梯度调和技术（如GradNorm、PCGrad）来减少任务间冲突

## 后续工作

1. 修复原始模型架构问题后重新进行完整实验
2. 实现梯度调和算法并评估其对TCI的影响
3. 探索任务特定的网络结构以减少共享层的冲突