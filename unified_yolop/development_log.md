# Unified YOLOP 开发日志

## 项目概述
本项目旨在统一和重构YOLOP系列模型（YOLOPv1、v2、v3、YOLOPx），提供统一的训练框架和接口。

## 已完成的工作

### 1. 项目结构搭建 ✓
- 创建了统一的项目目录结构
- 实现了模块化的代码组织（models、data、core、utils）

### 2. 统一接口实现 ✓
- **ModelFactory** (`models/factory.py`): 统一的模型创建工厂
- **UnifiedYOLOPDataset** (`data/dataset.py`): 统一的数据加载接口
- **UnifiedLoss** (`core/loss.py`): 统一的损失函数
- **Config** (`utils/config.py`): 统一的配置管理系统

### 3. YOLOPx 分割任务修复 ✓
**问题**：YOLOPx模型的分割损失为0，DA和Lane IoU都是0
**原因**：YOLOPx输出格式特殊，detection输出是tuple而非tensor
**解决方案**：
- 创建了 `models/yolopx_wrapper.py`，专门处理YOLOPx的输出格式
- 修改了 `ModelFactory` 使用新的wrapper
- 验证结果：分割任务现在正常工作，可以计算损失和指标

### 4. 冲突检测改进 ✓
- 创建了 `utils/conflict_improved.py` 改进梯度冲突计算
- 修复了检测任务冲突始终为0的问题
- 只计算共享层（backbone）的梯度，避免task-specific heads的影响

### 5. 指标计算系统 ✓
- 实现了 `core/metrics.py` 统一的评估指标
- **overall_score** 计算方法：
  - 检测F1分数权重：1.0
  - 可行驶区域IoU权重：0.5
  - 车道线IoU权重：0.5
  - 使用加权平均计算总分

### 6. 训练脚本 ✓
- `train.py`: 主训练脚本，支持所有模型
- `yolop_compare_experiment.py`: 批量对比实验脚本
- 支持TensorBoard和WandB日志记录
- 自动保存验证图像和模型检查点

## 技术细节

### Task Conflict 原理
- **本质**：计算不同任务损失梯度之间的余弦相似度
- **公式**：conflict = max(0, -cosine_similarity(grad1, grad2))
- **参考文献**：
  - "Gradient Surgery for Multi-Task Learning" (Yu et al., NeurIPS 2020)
  - "Multi-Task Learning as Multi-Objective Optimization" (Sener & Koltun, NeurIPS 2018)
- **意义**：当冲突值>0时，表示两个任务的梯度方向相反，存在优化冲突

### 验证图像说明
- **路径格式**：`/workspace/YOLOP_conflict/unified_yolop/runs/{model}_{timestamp}/val_epoch_{N}/`
- **图像内容**：
  - 绿色区域：可行驶区域分割预测
  - 红色/白色线条：车道线分割预测
  - 边界框：目标检测结果（如果有）

## 实验结果分析

### 2025-08-01 实验结果
1. **YOLOPx修复前**（20epoch实验）：
   - 分割任务完全失效（DA IoU=0, Lane IoU=0）
   - 只有检测任务工作

2. **YOLOPx修复后**（新实验）：
   - 分割任务正常工作
   - DA IoU: 0.9232, Lane IoU: 0.4605
   - overall_score: 0.6920

3. **YOLOPv2高冲突值**：
   - DA-LL冲突均值：0.011（比其他模型高100倍）
   - 可能原因：更大的模型容量（57.5M参数）导致任务间干扰增加

## 最新发现和修复（2025-08-01更新）

### 验证图像问题
**问题**：验证图像使用的是虚拟数据，不是真实的BDD100K道路场景
**原因**：
1. 数据集路径配置错误（使用了`images/100k/train`而非`images/train`）
2. 系统在使用`create_test_data.py`生成的虚拟数据
3. 验证图像保存函数缺少检测框绘制功能

**解决方案**：
1. 创建了`data/dataset_fixed.py`修复数据集路径
2. 创建了`utils/visualization.py`改进可视化功能：
   - 正确的图像反归一化
   - 检测框绘制
   - 半透明分割掩码叠加
3. 创建了`train_visualization_patch.py`更新训练脚本

## 正在进行的工作

### 1. 验证真实数据集
- 确认BDD100K数据集正确加载
- 运行小规模测试验证可视化效果

### 2. 损失记录功能增强
- 问题：训练损失曲线图为空（数据未正确记录）
- 需要修改实验脚本，收集并保存每个epoch的损失值

### 3. 验证YOLOPv2
- 由于没有官方源代码，主要目标是验证实现与论文描述一致
- 高冲突值可能是正常现象，需要进一步分析

## 待完成的工作

### 1. 完整的对比实验
- 在完整数据集上训练所有模型
- 收集详细的性能指标和冲突分析

### 2. 可视化改进
- 改进验证图像的可视化效果
- 添加检测框的标签显示

### 3. 文档完善
- 编写详细的使用文档
- 添加模型架构说明

### 4. 性能优化
- 优化数据加载速度
- 实现混合精度训练

## 重要文件路径

### 核心代码
- 模型工厂：`/workspace/YOLOP_conflict/unified_yolop/models/factory.py`
- YOLOPx包装器：`/workspace/YOLOP_conflict/unified_yolop/models/yolopx_wrapper.py`
- 统一损失：`/workspace/YOLOP_conflict/unified_yolop/core/loss.py`
- 冲突检测：`/workspace/YOLOP_conflict/unified_yolop/utils/conflict_improved.py`

### 实验结果
- 20epoch对比实验：`/workspace/YOLOP_conflict/unified_yolop/yolop_experiment_20250801_031826/`
- YOLOPx单独测试：`/workspace/YOLOP_conflict/unified_yolop/all_models_results_20250801_105106/`
- 验证图像：`/workspace/YOLOP_conflict/unified_yolop/runs/*/val_epoch_*/`

## 使用说明

### 训练单个模型
```bash
python train.py --model yolopx --epochs 20 --batch-size 16
```

### 运行对比实验
```bash
python yolop_compare_experiment.py --epochs 20 --parallel
```

### 快速测试修复
```bash
python test_fixes.py
```

## 注意事项

1. **数据集路径**：默认使用 `/workspace/bdd100k/yolop_train/`
2. **外部依赖**：
   - YOLOP_v1_official
   - YOLOP_v3_official
   - YOLOPX
3. **GPU内存**：建议使用至少16GB显存的GPU
4. **Python版本**：Python 3.8+

## 2025-08-01 最新更新 - 项目管理系统

### 新增统一项目管理系统
1. **ProjectManager类** (`utils/project_manager.py`)
   - 统一管理所有训练结果
   - 项目命名规则：`yolop_model_{num}_{timestamp}`
   - 所有结果保存在 `runs/project_name/` 下

2. **目录结构**
   ```
   runs/
   └── yolop_model_4_20250801_123456/
       ├── project_metadata.json
       ├── comparison_report.json
       ├── comparison_report.md
       ├── experiment.log
       ├── yolop_v1/
       │   ├── checkpoints/
       │   ├── logs/
       │   ├── visualizations/
       │   ├── config.json
       │   ├── results.json
       │   └── training.log
       ├── yolop_v2/
       ├── yolop_v3/
       └── yolopx/
   ```

3. **新训练脚本**
   - `train_unified.py`: 单模型训练，支持项目管理
   - `train_compare.py`: 批量训练和比较
   - `run_single_model.py`: 快速测试脚本

4. **使用方法**
   ```bash
   # 训练单个模型
   python train_unified.py --model yolopx --epochs 20
   
   # 训练并比较所有模型
   python train_compare.py --models all --epochs 20
   
   # 训练特定模型
   python train_compare.py --models yolop_v1 yolopx --epochs 20
   ```

## 2025-08-01 最新更新

### 代码清理和优化
1. **删除虚拟数据生成代码**
   - 删除了所有test_*.py, quick_*.py, debug_*.py等测试文件
   - 删除了create_test_data.py和demo_train.py

2. **数据集路径修正**
   - 固定路径：`/workspace/bdd100k/yolop_train/`
   - 训练图像：`/workspace/bdd100k/yolop_train/images/train/`
   - 验证图像：`/workspace/bdd100k/yolop_train/images/val/`
   - 使用assert而非try/except，确保错误立即暴露

3. **可视化改进**
   - 使用`visualization_dual.py`实现输入输出对比显示
   - 正确的ImageNet反归一化
   - 改进的检测框处理
   - 添加图例说明

4. **冲突检测增强**
   - 添加了`conflict_pcgrad.py`实现PCGrad论文的TCI指标
   - 记录余弦相似度（可以为负）
   - TCI = 1 - cos_sim
   - 冲突值 = max(0, -cos_sim)

5. **目录结构优化**
   - 删除了冗余的dataset_fixed.py
   - 删除了conflict_improved.py
   - 保持最小化的项目结构

## 更新时间
最后更新：2025-08-01