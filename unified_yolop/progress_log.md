# YOLOP Unified Training - 进度日志

## 项目状态
- **开始时间**: 2025-08-01
- **当前状态**: 代码整合完成，准备进行实验

## 已完成工作

### 1. 项目结构搭建 ✓
- 创建了模块化的代码组织
- 实现了统一的接口（ModelFactory, UnifiedDataset, UnifiedLoss）

### 2. 模型实现 ✓
- YOLOPv1: 通过外部库导入
- YOLOPv2: 自主实现（基于论文）
- YOLOPv3: 通过外部库导入
- YOLOPx: 通过外部库导入，添加了wrapper处理输出格式

### 3. 数据集配置 ✓
- 路径: `/workspace/bdd100k/yolop_train/`
- 训练集: `images/train/`
- 验证集: `images/val/`
- 包含检测、可行驶区域、车道线三个任务的标注

### 4. 关键修复 ✓
- **YOLOPx分割任务修复**: 创建了wrapper正确处理输出格式
- **冲突检测增强**: 实现了PCGrad论文的TCI指标
- **可视化改进**: 双图对比显示（输入vs输出）

### 5. 项目管理系统 ✓
- 统一的结果保存结构
- 项目命名: `yolop_model_{num}_{timestamp}`
- 自动生成比较报告（JSON和Markdown格式）

### 6. 训练脚本整合 ✓
- `run_experiment.py`: 统一入口，支持多种模式
  - single: 训练单个模型
  - compare: 比较指定模型
  - all: 比较所有模型
  - generate-scripts: 生成shell脚本

## 当前进度

### 代码状态
- ✓ 核心代码完成
- ✓ 数据集路径配置正确
- ✓ 项目管理系统实现
- ✓ 训练脚本整合
- ✓ Shell脚本生成

### 待测试
- [ ] 单模型训练流程
- [ ] 多模型比较流程
- [ ] 报告生成功能
- [ ] 可视化效果

## 下一步计划

### 1. 功能测试
```bash
# 快速测试（1 epoch）
./scripts/quick_test.sh

# 单模型完整训练
./scripts/train_yolopx.sh

# 所有模型比较
./scripts/train_compare_all.sh
```

### 2. 需要完善的部分

#### 2.1 数据增强
- 当前未实现数据增强
- 需要添加标准的增强策略（翻转、裁剪、颜色调整等）

#### 2.2 检测任务处理
- 检测输出后处理（NMS等）
- 检测框可视化需要进一步优化

#### 2.3 评估指标
- 添加更多评估指标（mAP for detection）
- 速度性能测试（FPS）

#### 2.4 模型优化
- 混合精度训练支持
- 模型量化和部署

### 3. 实验计划

#### 实验1: 基准性能测试
- 目标: 验证四个模型的基准性能
- 配置: 20 epochs, batch_size=8
- 预期时间: 4-6小时

#### 实验2: 任务冲突分析
- 目标: 分析多任务学习中的冲突
- 重点: TCI指标和性能相关性
- 输出: 冲突分析报告

#### 实验3: 消融实验
- 目标: 验证各组件的贡献
- 方法: 逐步移除/修改组件
- 输出: 性能影响分析

## 问题记录

### 已解决
1. ✓ YOLOPx分割损失为0 - 通过wrapper修复
2. ✓ 数据集路径错误 - 更正为正确路径
3. ✓ Python命令问题 - 使用sys.executable

### 待解决
1. 检测任务的评估指标（mAP）计算
2. 批量训练时的GPU内存管理
3. 大批量数据的训练稳定性

## 使用指南

### 快速开始
```bash
# 1. 测试环境
python3 run_experiment.py --mode single --model yolopx --epochs 1 --batch-size 4

# 2. 完整实验
python3 run_experiment.py --mode all --epochs 20 --batch-size 8

# 3. 查看结果
cd runs/yolop_model_4_*/
cat comparison_report.md
```

### 结果位置
- 项目目录: `runs/yolop_model_{num}_{timestamp}/`
- 模型权重: `{model}/checkpoints/best.pth`
- 可视化: `{model}/visualizations/epoch_{n}/`
- 比较报告: `comparison_report.md`

## 最新成功运行 (2025-08-01 16:16)

### 实验结果
1. **所有4个模型成功运行**
   - YOLOPv1: ✓ 0.7分钟
   - YOLOPv2: ✓ 1.2分钟 (模型较大)
   - YOLOPv3: ✓ 0.7分钟
   - YOLOPx: ✓ 0.7分钟

2. **解决的问题**
   - 修复了共享内存问题 (workers=0)
   - 修复了YOLOPv1的tuple检测输出问题
   - 修复了YOLOPv2的exp overflow问题
   - 添加了数据集大小控制功能

3. **实验设置**
   - 训练集: 200张图片
   - 验证集: 100张图片
   - Batch size: 2
   - Epochs: 1
   - Workers: 0

## 最新更新 (2025-08-01 16:00)

### 新增功能
1. **自定义数据集大小控制**
   - 添加 `--train-samples` 参数控制训练集大小
   - 添加 `--val-samples` 参数控制验证集大小
   - 示例: `--train-samples 50 --val-samples 20`
   - 优先级: 特定参数 > MAX_SAMPLES > 全部数据

### 参数传递链
- `run_experiment.py` → `train_compare.py` → `train_unified.py` → `UnifiedYOLOPDataset`
- 所有脚本都已更新支持新参数

## 最新测试结果 (2025-08-01 15:40)

### 测试状态
- ✓ 代码结构完成
- ✓ 项目管理系统工作
- ✗ 训练运行失败 - 共享内存不足

### 主要问题
1. **共享内存错误**
   - 错误: `DataLoader worker is killed by signal: Bus error`
   - 原因: Docker容器共享内存(shm)不足
   - 解决: 需要使用 `--workers 0` 或增加共享内存

### 已验证功能
1. **项目管理**
   - 自动创建项目目录 ✓
   - 命名规则正确: `yolop_model_4_20250801_154022` ✓
   - 生成比较报告 ✓

2. **模型加载**
   - YOLOPv1: 成功加载 (7,940,846 参数)
   - YOLOPv2: 待测试
   - YOLOPv3: 待测试  
   - YOLOPx: 成功加载 (7,940,846 参数)

3. **数据集**
   - 路径配置正确 ✓
   - 训练集: 70,000 图像
   - 验证集: 10,000 图像
   - Quick test模式: 100 图像

### 下一步行动
1. ✓ 修复共享内存问题 (使用 workers=0) - 已完成
2. ✓ 完成单模型测试 - 已完成
3. ✓ 添加数据集大小控制功能 - 已完成
4. 运行完整的4模型比较实验

## 已知问题
1. **项目管理器路径不一致**
   - train_unified.py创建了新的项目管理器而非使用传入的
   - 导致结果保存在不同的目录
   - 但不影响训练和模型保存

## 总结
- 代码成功运行，所有4个模型都可以训练
- 支持数据集大小控制
- 支持多模型对比
- 支持结果保存和报告生成
- 使用wandb记录训练过程

## 更新时间
最后更新: 2025-08-01 16:20