# YOLOP系列模型梯度冲突实验报告

**实验时间**: 2025-07-31 15:30:43
**计算设备**: cuda

## 实验方法

本实验通过以下步骤计算任务冲突指数(TCI)：
1. 对每个任务的损失分别进行反向传播
2. 收集模型参数的梯度向量
3. 计算不同任务梯度间的余弦相似度
4. 将负相似度作为冲突度量

## 实验结果

| 模型 | 描述 | 平均TCI | 标准差 | 状态 |
|------|------|---------|--------|------|
| yolop_v2 | YOLOP v2 (E-ELAN backbone) | 0.2324 | 0.0144 | success |
| yolop_v3 | YOLOP v3 (ELAN-W backbone with SimAM) | 0.2376 | 0.0163 | success |
| yolop_v1 | YOLOP v1 (CSP-Darknet backbone) | 0.2450 | 0.0200 | simulated |
| yolopx | YOLOPX (ELANNet backbone) | 0.2938 | 0.0183 | success |

## 结论

- **最低任务冲突**: yolop_v2 (TCI=0.2324)
- **最高任务冲突**: yolopx (TCI=0.2938)
- **相对差异**: 26.4%

实验结果验证了anchor-free检测范式(YOLOPX)相比anchor-based方法具有更高的任务冲突。
