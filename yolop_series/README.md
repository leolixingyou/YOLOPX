# YOLOP Series 代码结构详解

本文档详细说明yolop_series中每个目录和文件的用途。

## 目录结构概览

```
yolop_series/
├── cfgs/           # 配置文件目录
├── core/           # 核心功能模块
├── data/           # 数据集处理
├── engine/         # 训练引擎
├── models/         # 模型定义
├── mtl/            # 多任务学习模块
└── utils/          # 工具函数
```

---

## 1. cfgs - 配置文件目录

### 1.1 data/ - 数据集配置

#### bdd100k.yaml
- **用途**: BDD100K数据集的主要配置文件
- **关键配置**:
  - `NC: 1` - 单类检测（只检测车辆）
  - `NAMES: ['car']` - 类别名称
  - `NUM_SEG_CLASS: 2` - 分割类别数（背景+可行驶区域）
  - `NUMBER_IMAGE: 1000` - 限制使用的图片数量（调试用）
  - `IMAGE_SIZE: [384, 640]` - 输入图片尺寸
- **路径配置**:
  - DATAROOT: 原始图片路径
  - LABELROOT: 检测标签路径（JSON格式）
  - MASKROOT: 可行驶区域分割掩码
  - LANEROOT: 车道线分割掩码

#### dummy_bdd100k.yaml
- **用途**: 测试用的虚拟数据集配置
- **特点**: 指向生成的小规模测试数据

### 1.2 models/ - 模型配置

#### 模型配置文件说明

##### 1. yolop_v1.yaml vs yolop_v1_official.yaml
- **区别**:
  - `yolop_v1.yaml`: 简化版本，可能是早期实现
  - `yolop_v1_official.yaml`: 忠实复现官方YOLOPv1架构
- **主要差异**:
  ```yaml
  # yolop_v1.yaml
  prediction_heads:
    det_out_idx: 24
    da_seg_out_idx: 31
    ll_seg_out_idx: 32
  
  # yolop_v1_official.yaml
  model_family: yolop_v1_official  # 指定模型家族
  prediction_heads:
    det_out_idx: 24
    da_seg_out_idx: 33  # 不同的索引
    ll_seg_out_idx: 42  # 不同的索引
  ```
- **架构**: 都使用CSP-Darknet骨干网络

##### 2. yolop.yaml (YOLOPv2)
- **用途**: YOLOPv2模型配置
- **特点**: 使用E-ELAN骨干网络
- **改进**: 相比v1，优化了特征聚合策略

##### 3. yolop_v3_official.yaml
- **用途**: YOLOPv3官方架构
- **特点**: 
  - ELAN-W骨干网络（更宽的ELAN）
  - 集成SimAM注意力机制
  - 最低的任务冲突

##### 4. yolopx系列
- **yolopx.yaml**: 
  - 基础anchor-free版本
  - 使用ELANNet骨干 + YOLOXHead
  - 密集预测导致更高任务冲突

- **yolopx_tag系列**:
  - `yolopx_tag.yaml`: TAG（Task Adaptive Gradient）版本
  - `yolopx_tag_paper.yaml`: 论文版本配置
  - `yolopx_tag_simple.yaml`: 简化版TAG
  - `yolopx_tag_v2_compatible.yaml`: 与v2兼容的TAG版本

- **对比实验配置**:
  - `yolopx_v2_anchor_free.yaml`: anchor-free对比版本
  - `yolopx_v2_anchor_based.yaml`: anchor-based对比版本

### 1.3 训练配置

#### train_default.yaml
- **用途**: 默认训练配置
- **关键参数**:
  - 优化器: AdamW
  - 学习率: 0.001
  - 损失权重配置
  - CUDA设置

#### train_v2.yaml
- **用途**: v2版本专用训练配置
- **特点**: 针对v2架构优化的参数

#### train_temp.yaml
- **用途**: 临时测试配置

---

## 待添加内容

### 需要创建的数据配置

1. **多类检测配置** (bdd100k_multi_class.yaml)
   ```yaml
   NC: 10  # BDD100K的10个类别
   NAMES: ['person', 'rider', 'car', 'bus', 'truck', 
           'bike', 'motor', 'traffic light', 'traffic sign', 'train']
   ```

2. **单类检测配置** (bdd100k_single_class.yaml)
   ```yaml
   NC: 1  # 只检测车辆
   NAMES: ['car']
   ```

---

## 2. core - 核心功能模块

core目录包含了训练和推理的核心功能实现。

### 2.1 激活函数 (activations.py)
提供了多种高效的激活函数实现：
- **Swish**: x * sigmoid(x)，平滑的激活函数
- **Hardswish**: 移动端友好的Swish近似
- **MemoryEfficientSwish**: 内存优化版本，使用自定义梯度
- **Mish**: x * tanh(softplus(x))，更平滑的激活
- **MemoryEfficientMish**: 内存优化版本

### 2.2 损失函数

#### loss.py - 主损失函数
- **MultiHeadLoss**: 多任务损失的主类
  - 整合4个损失：检测损失、驾驶区域分割损失、车道线分割损失、Tversky损失
  - 支持损失权重调整（lambdas）
- **get_loss()**: 损失函数工厂方法
  - 检测：YOLOX_Loss（支持anchor-free）
  - 分割：BCEWithLogitsLoss（带正样本权重）
  - 车道线：可选FocalLoss增强
- **FocalLossSeg**: Focal Loss用于处理类别不平衡
- **TverskyLoss**: 用于分割任务，可调节假阳性和假阴性的权重

#### improved_loss.py
- 改进的损失权重设计：
  ```python
  det_weight = 0.05      # 降低检测权重
  da_seg_weight = 0.3    # 适中的驾驶区域权重
  ll_seg_weight = 0.4    # 提高车道线权重
  ll_iou_weight = 0.3    # 提高IoU权重
  ```
- 更好地平衡多任务学习

#### improved_tag_loss.py
- TAG（Task Adaptive Gradient）版本的损失
- 实现任务自适应的梯度调整

#### unified_loss.py
- 统一的损失函数接口
- 支持不同模型使用相同的损失计算流程

### 2.3 任务冲突处理 (conflict_methods.py)

实现了多种多任务学习优化方法：

#### GradNormBalancer
- **原理**: 基于GradNorm论文，自适应调整任务权重
- **核心思想**: 
  - 监控每个任务的学习速度
  - 动态调整权重使所有任务以相似速度学习
  - 通过梯度范数来衡量任务难度
- **参数**:
  - `alpha`: 恢复力强度（越高越激进）
  - `update_freq`: 权重更新频率
- **功能**:
  - 跟踪损失历史
  - 计算相对学习速度
  - 可视化权重变化

### 2.4 评估函数 (evaluate.py)
- **ap_per_class()**: 计算每个类别的平均精度(AP)
- **compute_ap()**: 计算精确率-召回率曲线下的面积
- **fitness()**: 模型性能的综合评分
  - 权重：[P, R, mAP@0.5, mAP@0.5:0.95] = [0.0, 0.0, 0.1, 0.9]
  - 主要关注mAP@0.5:0.95

### 2.5 工具函数

#### general.py
- 通用工具函数
- 数据处理辅助函数

#### function.py
- 训练流程相关函数
- 模型保存/加载

#### postprocess.py
- **build_targets()**: 为YOLO检测头构建训练目标
  - 匹配anchor和真实框
  - 计算偏移量
  - 支持多尺度预测
- 后处理流程：NMS、聚类等

### 2.6 关键设计

1. **多任务损失平衡**：
   - 基础版本：固定权重
   - 改进版本：调整权重比例
   - GradNorm：动态自适应权重

2. **内存优化**：
   - 自定义激活函数梯度计算
   - 减少中间变量存储

3. **灵活性**：
   - 支持不同的检测头（anchor-based/free）
   - 可配置的损失组合
   - 模块化设计便于扩展

---

## 3. data - 数据集处理

data目录负责数据集的加载、预处理和数据增强。

### 3.1 数据集类层次结构

#### AutoDriveDataset (base类)
- **AutoDriveDataset.py**: 旧版本基类（V1遗留）
- **autodrive_dataset.py**: 简化版本，移除了外部依赖
- **unified_dataset.py**: 统一的数据集实现（推荐使用）
  - 统一了数据加载逻辑
  - 支持多任务：检测、驾驶区域分割、车道线分割
  - 内置letterbox调整和批处理

#### BddDataset实现
- **bdd.py**: 原始BDD100K数据集实现
  - 支持单类/多类检测切换
  - 使用JSON格式标注
  - `BddDataset`: 硬编码200张训练图片
  - `BddDataset_refactor`: 使用配置文件中的NUMBER_IMAGE

- **bdd_dataset.py**: autodrive_dataset的包装版本

- **DemoDataset.py**: 演示数据集，用于测试

### 3.2 数据转换和工具

#### convert.py
- **坐标转换**: bbox格式转换为YOLO格式(中心点+宽高)
- **类别映射**:
  ```python
  # 多类别映射（13类）
  id_dict = {'person': 0, 'rider': 1, 'car': 2, 'bus': 3, 'truck': 4, 
             'bike': 5, 'motor': 6, 'tl_green': 7, 'tl_red': 8, 
             'tl_yellow': 9, 'tl_none': 10, 'traffic sign': 11, 'train': 12}
  
  # 单类别映射（4类车辆）
  id_dict_single = {'car': 0, 'bus': 1, 'truck': 2, 'train': 3}
  ```

#### transforms.py
- **letterbox()**: 保持宽高比的图片缩放和填充
- **augment_hsv()**: HSV色彩空间增强
- **random_perspective()**: 透视变换增强（旋转、缩放、剪切、平移）

### 3.3 数据加载流程

1. **图片路径构建**:
   - 原始图片: `DATAROOT/train/*.jpg`
   - 检测标签: `LABELROOT/train/*.json`
   - 驾驶区域: `MASKROOT/train/*.png`
   - 车道线: `LANEROOT/train/*.png`

2. **标签处理**:
   - JSON解析BDD100K格式
   - 转换为YOLO格式(归一化的中心点坐标)
   - 支持交通灯颜色属性

3. **批处理collate_fn**:
   - 为每个样本添加批次索引
   - 处理空标签情况
   - 返回格式: (images, [det_labels, seg_masks, lane_masks], paths)

---

## 4. models - 模型定义

models目录包含所有模型架构的实现。

### 4.1 模型构建器 (builder.py)

**MCnetFromYAML类**:
- 从YAML配置文件构建模型
- 支持三种模型家族:
  - `yolop_v1_official`: V1官方架构
  - `yolop_v3_official`: V3官方架构  
  - 默认: YOLOPX v2架构
- 动态加载对应的模块集合
- 管理预测头索引（检测、驾驶区域、车道线）

### 4.2 通用模块 (common_modules.py)

#### 基础组件
- **Conv**: 卷积+BN+激活的基础块
- **RepConv**: 重参数化卷积（训练时3x3+1x1，推理时融合）
- **Bottleneck**: 残差瓶颈块
- **C3/CSPLayer**: CSP结构变体

#### ELAN系列
- **Conv_ELANBlock**: ELAN基础块
- **ELANBlock**: 标准ELAN块（多分支聚合）
- **ELANBlock_Head**: 用于检测头的ELAN块
- **ELANNet**: ELAN骨干网络
- **PaFPNELAN**: 基于ELAN的PaFPN

#### 注意力机制
- **SimAM**: 无参数的简单注意力模块
- **SPPCSPC**: 空间金字塔池化的CSP版本
- **PSA_p**: 像素级注意力模块

#### 任务头
- **seg_head**: 分割头（sigmoid/softmax）
- **MergeBlock**: 特征融合块
- **FPN_C2/C3/C4**: 特征金字塔选择器

### 4.3 检测头 (heads/)

#### yolop_head.py
- **Detect类**: Anchor-based检测头
- 支持多尺度预测（3个level）
- 包含anchor生成和解码逻辑

#### yolox_head.py  
- **YOLOXHead**: Anchor-free检测头
- 解耦的分类和回归分支
- 包含:
  - stems: 降维1x1卷积
  - cls_convs/reg_convs: 分类/回归卷积
  - cls_preds/reg_preds/obj_preds: 预测层
- 支持训练时的标签分配和推理时的解码

### 4.4 损失函数 (losses/)

- **YOLOX_Loss.py**: Anchor-free检测损失
- **focal_loss.py**: Focal Loss实现
- **tversky_loss.py**: Tversky Loss（分割任务）
- **general.py**: 通用损失函数
- **simple_loss.py**: 简化的多任务损失

### 4.5 模型家族特定模块

#### yolop_v1_official_modules/
- V1专用的CSP-Darknet组件
- Focus、SPP等模块

#### yolop_v3_official_modules/
- V3专用的ELAN-W组件
- 集成SimAM注意力

---

## 5. engine - 训练引擎

engine目录包含训练和验证的核心逻辑。

### 5.1 训练器 (trainer.py)

**Trainer类**:
- **训练循环管理**:
  - 批次迭代和进度条
  - 混合精度训练（AMP）
  - 梯度累积和缩放

- **学习率调度**:
  - Warmup阶段的线性增长
  - Cosine退火调度
  - 不同参数组的独立调度

- **多任务损失处理**:
  - 支持固定权重或动态权重（通过solver）
  - 任务冲突检测（TCI计算）
  - 实时损失监控

- **集成功能**:
  - WandB日志记录
  - 梯度冲突解决器（GradNorm等）

### 5.2 验证器 (validator.py)

**Validator类**:
- **评估指标**:
  - 检测：mAP、精确率、召回率
  - 分割：像素精度、mIoU
  - 推理时间统计

- **可视化**:
  - 检测框绘制
  - 分割掩码叠加
  - 结果保存和上传

- **批处理**:
  - NMS后处理
  - 多尺度评估支持

---

## 6. mtl - 多任务学习模块

mtl目录实现了各种多任务学习优化方法。

### 6.1 梯度级别方法 (grad_solvers.py)

**FixedGradientConflictSolver类**:

#### 支持的方法
1. **GradNorm**:
   - 动态调整任务权重
   - 基于学习速度平衡
   - 参数：alpha（恢复力）、update_freq（更新频率）

2. **PCGrad**:
   - 投影冲突梯度
   - 保证梯度方向不冲突
   - 减少负迁移

3. **CAGrad**:
   - 基于优化的梯度调整
   - 参数：c（正则化系数）

4. **TAG**:
   - 架构级别的解决方案
   - 简单的梯度求和

### 6.2 TAG模块 (tag_module.py)

**TaskAdaptiveAttentionGenerator**:
- **双重注意力机制**:
  - αt：任务特定的通道注意力
  - β：任务通用的空间注意力
  - ht = αt · h + β

- **组件**:
  - 任务通道注意力（每个任务一个）
  - 空间注意力（共享）
  - 特征融合层

**辅助类**:
- **TAGSelect**: 选择特定任务的特征
- **TAGFusionLayer**: 融合TAG特征和骨干特征

### 6.3 其他MTL方法

- **feature_solvers.py**: 特征级别的MTL方法
- **heuristic_solver.py**: 启发式的任务平衡方法

---

## 7. utils - 工具函数

utils目录提供各种辅助功能。

### 7.1 核心工具

#### general.py
- **数据处理**: 标签格式转换、bbox操作
- **模型工具**: 参数统计、模型加载/保存
- **训练辅助**: EMA、梯度裁剪、检查点管理

#### torch_utils.py
- **初始化**: 权重初始化策略
- **优化**: 模型融合、剪枝
- **设备管理**: GPU选择、内存管理

### 7.2 评估和可视化

#### metrics.py
- **检测指标**:
  - AP计算（ap_per_class）
  - 混淆矩阵（ConfusionMatrix）
  - F1分数

- **分割指标**:
  - SegmentationMetric类
  - 像素精度、IoU、mIoU

#### plots.py / plot.py
- **可视化函数**:
  - 训练曲线绘制
  - 检测结果可视化
  - 分割掩码渲染
  - 类别颜色映射

### 7.3 数据处理

#### augmentations.py
- 数据增强管道
- 支持多任务的同步增强

#### autoanchor.py
- Anchor尺寸自动优化
- K-means聚类分析

#### split_dataset.py
- 数据集划分工具
- 训练/验证/测试集生成

### 7.4 日志和监控

#### logger.py
- 统一的日志接口
- 支持控制台和文件输出
- WandB集成

---

## 总结

yolop_series是一个完整的多任务学习框架，专门针对自动驾驶感知任务：

1. **模块化设计**: 各组件解耦，易于扩展
2. **多模型支持**: 统一接口支持V1/V2/V3/YOLOPX
3. **任务冲突处理**: 多种MTL优化方法
4. **完整的训练流程**: 从数据加载到模型评估
5. **灵活的配置**: YAML配置驱动的架构

该框架特别适合研究多任务学习中的任务冲突问题，以及在自动驾驶场景下的实际应用。