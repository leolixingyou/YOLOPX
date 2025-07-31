# YOLOPX 实验目录说明

## 如何运行实验

### 主要实验文件

要运行YOLOP系列模型的任务冲突对比实验，请使用：

```bash
cd /workspace/YOLOPX
python3 yolop_series/unified_experiment_v2.py
```

这个脚本会：
1. 对比所有YOLOP模型（v1, v2, v3, YOLOPX）的任务冲突强度(TCI)
2. 生成JSON格式的详细结果
3. 创建Markdown格式的可视化报告

### 实验结果位置

所有结果保存在：
- **JSON数据**: `/workspace/YOLOPX/experiments/base_models_comparison/results/yolop_series_comparison_*.json`
- **可视化报告**: `/workspace/YOLOPX/experiments/base_models_comparison/results/visual_report_*.md`

### 最新实验结果

最新实验（2025-07-31）显示：
- **YOLOP v3**: 最低TCI (0.2235) - ELAN-W backbone + SimAM注意力
- **YOLOP v2**: TCI 0.2295 - E-ELAN backbone
- **YOLOP v1**: TCI 0.2405 - CSP-Darknet backbone  
- **YOLOPX**: 最高TCI (0.2865) - ELANNet backbone + anchor-free检测

**关键发现**: Anchor-free (YOLOPX) 比最佳anchor-based模型 (v3) 的任务冲突高约28%。

### 其他实验文件说明

`base_models_comparison/` 目录下的其他文件大多是开发过程中的中间版本，保留它们是为了：
- 记录实验迭代过程
- 提供不同实验方法的参考
- 便于后续深入研究时回溯

如果只需要运行对比实验，使用 `unified_experiment_v2.py` 即可。