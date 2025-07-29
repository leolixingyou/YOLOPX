# YOLOPX v2 测试报告

## 1. 设计

本次测试旨在对 `YOLOPX v2` 的核心模型 `yolopx` 进行初步的功能性和性能验证。测试将遵循以下设计：

- **模型**: 使用 `cfgs/models/yolopx.yaml` 中定义的 `YOLOPX` 架构。
- **数据集**: 使用 `cfgs/data/bdd100k.yaml` 配置的 BDD100K 数据集。
- **训练配置**: 基于 `cfgs/train_default.yaml` 的默认参数。
- **多任务学习 (MTL) 策略**: 采用 `original` 基线方法，不引入额外的梯度优化策略。
- **训练周期**: 为快速验证，将训练周期临时设置为 5 个 epoch。

## 2. 目的

本次测试旨在验证 YOLOPX v2 版本的有效性、稳定性和性能，确保其在各项感知任务（如目标检测、可行驶区域分割、车道线检测）上达到预期标准。

## 3. 经过

### 3.1. 环境准备

1.  **修改训练周期**:
    *   **文件**: `cfgs/train_default.yaml`
    *   **操作**: 将 `END_EPOCH` 参数从 `200` 修改为 `5`，以进行快速的功能验证。
2.  **修复配置缺失**:
    *   **文件**: `cfgs/train_default.yaml`
    *   **问题**: 脚本因缺少 `LOG_DIR` 配置项而失败。
    *   **操作**: 添加 `LOG_DIR: 'runs/'` 到配置文件中。

### 3.2. 执行训练

使用以下命令启动训练脚本：

```bash
python3 tools/train.py \
    --model-cfg cfgs/models/yolopx.yaml \
    --data-cfg cfgs/data/bdd100k.yaml \
    --train-cfg cfgs/train_default.yaml \
    --mtl-method original
```

## 4. 结果

(此处将汇总和格式化测试的量化结果，参考 v1/test_log.out 的格式。)

## 5. 总结

(此处将对整个测试过程和结果进行分析和总结。)
