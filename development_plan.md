# 统一开发计划：YOLOP 任务冲突研究

## 1. 导言与目标

本计划旨在综合我们之前的所有讨论（`discuss_gemini.md`, `discuss_claude.md`）以及对 `unified_yolop` 代码库的深入分析，为下一阶段的研究与开发工作制定一份清晰、统一且可执行的行动路线图。

**核心目标：**

1.  **系统性研究**：在严格控制变量的条件下，系统性地实现、测试和分析多种多任务学习（MTL）优化策略对 YOLOP 系列模型的影响。
2.  **代码库整合**：将所有开发力量集中于 `unified_yolop`，并妥善处理遗留的 `yolop_series` 代码库，确保项目的清晰度和可维护性。
3.  **高效实验**：构建一个灵活、可扩展的实验框架，能够以“即插即用”的方式切换 MTL 策略，并为未来应对训练效率挑战打下基础。

---

## 2. 代码库处理策略

根据我们的共识，现有代码库的处理方式如下：

*   **`yolop_series`**
    *   **决策**：**正式归档 (Archive)**。
    *   **行动**：我将在此目录中创建一个 `ARCHIVED_README.md` 文件。该文件会明确指出：此目录下的代码是早期探索性实现，已被 `unified_yolop` 项目取代，不应再用于新的开发或严谨的实验，以防未来的研究者产生混淆。

*   **`unified_yolop`**
    *   **决策**：**作为未来所有开发的唯一代码库**。
    *   **行动**：所有新的 MTL 策略实现、实验流程修改和分析都将在此项目内进行。

---

## 3. `unified_yolop` 开发计划：分阶段实施

我们将采用分阶段的方法，确保每一步都稳固可靠。

### **阶段 0: 基础构建 - MTL 策略模块与接口设计**

此阶段的目标是搭建一个优雅、可扩展的框架，而不是直接修改训练脚本。

1.  **创建模块目录**：在 `unified_yolop/` 下创建新目录 `mtl_strategies`。
2.  **定义抽象基类**：在 `unified_yolop/mtl_strategies/base.py` 中，定义一个接口（抽象基类）`MTLStrategy`。所有具体的优化算法都将继承此类。
3.  **创建工厂函数**：在 `unified_yolop/mtl_strategies/__init__.py` 中，创建一个工厂函数 `create_mtl_strategy()`，它能根据输入的字符串名称（如 `'pcgrad'`）返回对应的策略实例。

### **阶段 1: 框架集成 - 对接现有训练流程**

此阶段将 MTL 框架无缝接入现有代码。

1.  **修改启动器 `run_experiment.py`**:
    *   为其参数解析器增加一个新的命令行参数：`--mtl-strategy`，默认值为 `'original'`。
    *   将此参数的值传递给它调用的 `train_unified.py` 子进程。

2.  **修改训练脚本 `train_unified.py`**:
    *   接收 `--mtl-strategy` 参数。
    *   在 `main()` 函数中，调用阶段 0 创建的工厂函数 `create_mtl_strategy()` 来实例化选择的策略对象。
    *   将这个策略对象传递给核心的 `train_epoch()` 函数。

### **阶段 2: 实现基线与首个优化策略**

此阶段验证整个框架的有效性。

1.  **实现 `OriginalStrategy`**：在 `mtl_strategies/original.py` 中实现。它的逻辑就是当前的做法：简单地将所有任务的损失相加，然后调用 `backward()`。
2.  **实现 `PCGradStrategy`**：在 `mtl_strategies/pcgrad.py` 中实现。这将是第一个真正的梯度优化策略，需要仔细处理梯度的计算和投影。
3.  **测试与验证**：使用 `--quick-test` 模式，分别用 `--mtl-strategy original` 和 `--mtl-strategy pcgrad` 运行训练，确保两种模式都能正常工作，且 PCGrad 的逻辑符合预期。

### **阶段 3: 扩充策略库 **

逐一实现并验证其他 MTL 优化策略。

1.  **实现 `CAGradStrategy`**。
2.  **实现 `GradNormStrategy`** (注意：此策略可能需要修改接口，因为它需要更新任务权重)。
3.  **实现其他**您感兴趣的策略 (如 TAG, MDO)。

### **阶段 4: 全面实验与分析 (持续进行)**

在框架稳定后，系统性地开展实验。

1.  **设计实验**：使用 `run_experiment.py` 的不同模式，组合不同的模型（`--model`）和 MTL 策略（`--mtl-strategy`）进行对比实验。
2.  **分析结果**：根据我们之前讨论的维度（收敛速度、最终性能、稳定性、任务冲突指标）进行深入分析，并撰写报告。

---

## 4. 核心技术设计：统一的 MTL 策略处理器

为了响应您将 `if/elif` 重构为统一函数的需求，我们将采用基于策略模式（Strategy Pattern）的设计。

**1. 抽象基类 (`mtl_strategies/base.py`)**

```python
from abc import ABC, abstractmethod

class MTLStrategy(ABC):
    """多任务学习优化策略的抽象基类"""
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    @abstractmethod
    def backward(self, losses: dict, **kwargs) -> None:
        """
        根据策略计算并应用梯度。
        
        Args:
            losses (dict): 一个包含各任务损失的字典, e.g., 
                           {'detection': loss_det, 'da_seg': loss_ll, ...}
        """
        pass

    def step(self):
        """执行优化器的 step。大多数策略使用默认即可。"""
        self.optimizer.step()

```

**2. `train_epoch` 函数的改造 (`train_unified.py`)**

当前的 `train_epoch` 函数将被大幅简化和清晰化。

```python
# 伪代码，展示核心修改
def train_epoch(model, train_loader, criterion, optimizer, device, 
                mtl_strategy: MTLStrategy, # 接收策略对象
                conflict_detector, epoch, logger, writer=None):
    # ... (循环开始)
    
    # ... (数据准备)
    
    # 前向传播
    outputs = model(images)
    
    # 计算损失
    losses = criterion(outputs, targets)
    
    # 分析冲突 (此部分逻辑不变，仅用于分析和记录)
    conflicts = conflict_detector.detect_conflicts(model, losses)
    
    # --- 核心修改 --- #
    # 旧逻辑:
    # total_loss_batch = sum(losses.values())
    # optimizer.zero_grad()
    # total_loss_batch.backward()
    
    # 新的、统一的调用方式:
    optimizer.zero_grad()
    mtl_strategy.backward(losses) # 将梯度计算和反向传播的复杂性委托给策略对象
    # --- 修改结束 --- #
    
    # 梯度裁剪 (可选，但建议保留)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    
    # 更新模型权重
    mtl_strategy.step() # 使用策略的 step 方法，以支持需要特殊处理的优化器
    
    # ... (日志记录等)
```

这种设计将不同 MTL 策略的复杂实现完全封装在各自的类中，使得主训练循环保持干净、稳定，并且极易扩展。

---

## 5. 下一步行动

我建议我们按照此计划，从**阶段 0** 和 **处理 `yolop_series`** 开始。如果您同意此开发计划，我将立即着手执行以下两项任务：

1.  在 `YOLOP_conflict/yolop_series` 目录下创建 `ARCHIVED_README.md` 文件。
2.  在 `YOLOP_conflict/unified_yolop` 目录下创建 `mtl_strategies` 文件夹和 `base.py` 文件，并定义好 `MTLStrategy` 抽象基类。

期待您的确认。
