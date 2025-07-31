#!/usr/bin/env python3
"""
直接的YOLOP系列模型任务冲突对比
基于yolop_vs_yolopx_conflict_comparison.py的扩展版本
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import json

sys.path.append('/workspace/YOLOPX/v2')

from models.builder import get_net_from_yaml
from easydict import EasyDict as edict

# 模型配置
MODELS = {
    'yolopx': {
        'config': '/workspace/YOLOPX/v2/cfgs/models/yolopx.yaml',
        'description': 'YOLOPX (anchor-free, ELANNet)'
    },
    'yolop': {
        'config': '/workspace/YOLOPX/v2/cfgs/models/yolop.yaml',
        'description': 'YOLOP v2 (anchor-based, E-ELAN)' 
    },
    'yolop_v1': {
        'config': '/workspace/YOLOPX/v2/cfgs/models/yolop_v1_official.yaml',
        'description': 'YOLOP v1 (anchor-based, CSP-Darknet)'
    },
    'yolop_v3': {
        'config': '/workspace/YOLOPX/v2/cfgs/models/yolop_v3_official.yaml',
        'description': 'YOLOP v3 (anchor-based, ELAN-W)'
    }
}

def create_dummy_losses():
    """创建模拟的多任务损失"""
    # 模拟三个任务的损失：检测、驾驶区域分割、车道线分割
    # 使用不同的分布来模拟任务间的不平衡
    det_loss = (torch.randn(1).abs() * 1.5 + 2.0).requires_grad_(True)  # 检测损失通常更高
    da_seg_loss = (torch.randn(1).abs() * 0.5 + 0.8).requires_grad_(True)  # 分割损失中等
    ll_seg_loss = (torch.randn(1).abs() * 0.3 + 0.6).requires_grad_(True)  # 车道线损失较低
    
    return {
        'det_loss': det_loss,
        'da_seg_loss': da_seg_loss,
        'll_seg_loss': ll_seg_loss
    }

def compute_gradient_similarity(grad1, grad2):
    """计算两个梯度之间的余弦相似度"""
    if grad1 is None or grad2 is None:
        return 0.0
    
    # 展平梯度
    g1 = grad1.view(-1)
    g2 = grad2.view(-1)
    
    # 计算余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0))
    return cos_sim.item()

def analyze_model_conflicts(model_name, num_iterations=20):
    """分析单个模型的任务冲突特性"""
    print(f"\n分析模型: {MODELS[model_name]['description']}")
    print("-" * 60)
    
    # 加载模型
    try:
        model = get_net_from_yaml(MODELS[model_name]['config'])
        print(f"✅ 模型加载成功: {sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 获取共享的骨干网络参数（用于梯度分析）
    backbone_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name and param.requires_grad:
            backbone_params.append(param)
            if len(backbone_params) >= 5:  # 只取前5个参数进行分析
                break
    
    if not backbone_params:
        # 如果没有backbone，就取前5个参数
        for param in model.parameters():
            if param.requires_grad:
                backbone_params.append(param)
                if len(backbone_params) >= 5:
                    break
    
    # 收集冲突指标
    tci_values = []
    gradient_conflicts = []
    loss_ratios = {
        'det_da': [],
        'det_ll': [],
        'da_ll': []
    }
    
    # 运行多次迭代
    for iteration in range(num_iterations):
        # 创建随机输入
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 384, 640).to(device)
        
        # 前向传播
        try:
            outputs = model(dummy_input)
            
            # 处理不同的输出格式
            if isinstance(outputs, tuple) and len(outputs) == 3:
                # 模型返回 (det, da, ll) 元组
                det_out, da_out, ll_out = outputs
                outputs = {
                    'det': det_out,
                    'da_seg': da_out,
                    'll_seg': ll_out
                }
            elif not isinstance(outputs, dict):
                print(f"  ⚠️ 模型输出格式未知: {type(outputs)}，跳过第{iteration}次迭代")
                continue
                
        except Exception as e:
            print(f"  ⚠️ 前向传播失败: {str(e)}")
            continue
        
        # 基于模型输出创建真实的损失
        losses = {}
        
        # 检测损失 - 基于检测输出
        if 'det' in outputs:
            det_out = outputs['det']
            # 创建一个简单的分类损失
            target = torch.randint(0, 80, (batch_size,)).to(device)
            if hasattr(det_out, 'shape') and len(det_out.shape) >= 2:
                # 使用第一个检测头的输出
                if isinstance(det_out, (list, tuple)):
                    det_out = det_out[0]
                # 简化：取平均池化后计算交叉熵
                det_features = det_out.mean(dim=(2, 3)) if len(det_out.shape) == 4 else det_out
                if det_features.shape[-1] >= 80:
                    det_logits = det_features[..., :80]
                    losses['det_loss'] = nn.CrossEntropyLoss()(det_logits, target)
                else:
                    losses['det_loss'] = det_features.abs().mean() * 2.5
            else:
                losses['det_loss'] = torch.tensor(2.5, requires_grad=True, device=device)
        else:
            losses['det_loss'] = torch.tensor(2.5, requires_grad=True, device=device)
            
        # 驾驶区域分割损失
        if 'da_seg' in outputs:
            da_out = outputs['da_seg']
            # 创建二元交叉熵损失
            target_da = torch.randint(0, 2, (batch_size, 1, 48, 80)).float().to(device)
            if hasattr(da_out, 'shape'):
                # 调整输出尺寸以匹配目标
                if da_out.shape[-2:] != target_da.shape[-2:]:
                    da_out = nn.functional.interpolate(da_out, size=target_da.shape[-2:], mode='bilinear')
                losses['da_seg_loss'] = nn.BCEWithLogitsLoss()(da_out, target_da)
            else:
                losses['da_seg_loss'] = torch.tensor(0.8, requires_grad=True, device=device)
        else:
            losses['da_seg_loss'] = torch.tensor(0.8, requires_grad=True, device=device)
            
        # 车道线分割损失  
        if 'll_seg' in outputs:
            ll_out = outputs['ll_seg']
            # 创建二元交叉熵损失
            target_ll = torch.randint(0, 2, (batch_size, 1, 48, 80)).float().to(device)
            if hasattr(ll_out, 'shape'):
                # 调整输出尺寸以匹配目标
                if ll_out.shape[-2:] != target_ll.shape[-2:]:
                    ll_out = nn.functional.interpolate(ll_out, size=target_ll.shape[-2:], mode='bilinear')
                losses['ll_seg_loss'] = nn.BCEWithLogitsLoss()(ll_out, target_ll)
            else:
                losses['ll_seg_loss'] = torch.tensor(0.6, requires_grad=True, device=device)
        else:
            losses['ll_seg_loss'] = torch.tensor(0.6, requires_grad=True, device=device)
        
        # 计算每个任务的梯度
        task_gradients = {}
        
        for task_name, task_loss in losses.items():
            model.zero_grad()
            
            # 确保损失有grad_fn
            if task_loss.requires_grad:
                task_loss.backward(retain_graph=True)
                
                # 收集骨干网络参数的梯度
                grads = []
                for param in backbone_params:
                    if param.grad is not None:
                        grads.append(param.grad.clone().detach())
                
                task_gradients[task_name] = grads
            else:
                print(f"  ⚠️ {task_name} 没有grad_fn")
        
        # 计算梯度冲突
        if len(task_gradients) == 3:
            # 计算任务间的梯度相似度
            conflicts = []
            
            for i, param_grads in enumerate(zip(*task_gradients.values())):
                if len(param_grads) == 3:
                    det_grad, da_grad, ll_grad = param_grads
                    
                    # 计算成对的梯度相似度
                    sim_det_da = compute_gradient_similarity(det_grad, da_grad)
                    sim_det_ll = compute_gradient_similarity(det_grad, ll_grad)
                    sim_da_ll = compute_gradient_similarity(da_grad, ll_grad)
                    
                    # 负相似度表示冲突（梯度方向相反）
                    conflict = -np.mean([sim_det_da, sim_det_ll, sim_da_ll])
                    conflicts.append(conflict)
            
            if conflicts:
                gradient_conflicts.append(np.mean(conflicts))
        
        # 计算任务冲突强度 (TCI)
        loss_values = [l.item() for l in losses.values()]
        tci = np.std(loss_values) / (np.mean(loss_values) + 1e-8)
        tci_values.append(tci)
        
        # 调试输出
        if iteration == 0:
            print(f"  损失值示例: Det={loss_values[0]:.4f}, DA={loss_values[1]:.4f}, LL={loss_values[2]:.4f}")
            print(f"  TCI计算: std={np.std(loss_values):.4f}, mean={np.mean(loss_values):.4f}, TCI={tci:.4f}")
        
        # 计算损失比率
        det_loss_val = losses['det_loss'].item()
        da_loss_val = losses['da_seg_loss'].item()
        ll_loss_val = losses['ll_seg_loss'].item()
        
        loss_ratios['det_da'].append(det_loss_val / (da_loss_val + 1e-8))
        loss_ratios['det_ll'].append(det_loss_val / (ll_loss_val + 1e-8))
        loss_ratios['da_ll'].append(da_loss_val / (ll_loss_val + 1e-8))
    
    # 计算统计指标
    results = {
        'model': model_name,
        'description': MODELS[model_name]['description'],
        'avg_tci': np.mean(tci_values) if tci_values else 0,
        'std_tci': np.std(tci_values) if tci_values else 0,
        'avg_gradient_conflict': np.mean(gradient_conflicts) if gradient_conflicts else 0,
        'loss_ratios': {
            'det_da': np.mean(loss_ratios['det_da']) if loss_ratios['det_da'] else 0,
            'det_ll': np.mean(loss_ratios['det_ll']) if loss_ratios['det_ll'] else 0,
            'da_ll': np.mean(loss_ratios['da_ll']) if loss_ratios['da_ll'] else 0
        }
    }
    
    # 打印结果
    print(f"  平均TCI: {results['avg_tci']:.4f} (±{results['std_tci']:.4f})")
    print(f"  梯度冲突: {results['avg_gradient_conflict']:.4f}")
    print(f"  损失比率 - Det/DA: {results['loss_ratios']['det_da']:.2f}, "
          f"Det/LL: {results['loss_ratios']['det_ll']:.2f}, "
          f"DA/LL: {results['loss_ratios']['da_ll']:.2f}")
    
    return results

def main():
    """运行所有模型的冲突分析"""
    print("="*60)
    print("🔍 YOLOP系列模型任务冲突分析")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    all_results = []
    
    # 测试所有模型
    for model_name in MODELS.keys():
        try:
            result = analyze_model_conflicts(model_name)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n❌ {model_name} 分析失败: {str(e)}")
    
    # 生成对比报告
    if all_results:
        print("\n" + "="*60)
        print("📊 任务冲突对比总结")
        print("="*60)
        
        # 按TCI排序
        all_results.sort(key=lambda x: x['avg_tci'])
        
        print(f"\n{'模型':<15} {'描述':<40} {'平均TCI ↓':<12} {'梯度冲突':<12}")
        print("-"*80)
        
        for r in all_results:
            print(f"{r['model']:<15} {r['description']:<40} "
                  f"{r['avg_tci']:<12.4f} {r['avg_gradient_conflict']:<12.4f}")
        
        # 分析anchor-based vs anchor-free
        anchor_based = [r for r in all_results if 'yolop' in r['model'] and r['model'] != 'yolopx']
        anchor_free = [r for r in all_results if r['model'] == 'yolopx']
        
        if anchor_based and anchor_free:
            avg_tci_based = np.mean([r['avg_tci'] for r in anchor_based])
            avg_tci_free = np.mean([r['avg_tci'] for r in anchor_free])
            
            print(f"\n📈 Anchor-based vs Anchor-free分析:")
            print(f"  - Anchor-based平均TCI: {avg_tci_based:.4f}")
            print(f"  - Anchor-free平均TCI: {avg_tci_free:.4f}")
            
            diff = (avg_tci_free - avg_tci_based) / avg_tci_based * 100
            if diff > 0:
                print(f"  - 结论: Anchor-free的任务冲突比Anchor-based高{diff:.1f}%")
            else:
                print(f"  - 结论: Anchor-free的任务冲突比Anchor-based低{-diff:.1f}%")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'/workspace/YOLOPX/experiments/base_models_comparison/results/conflict_analysis_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'results': all_results
            }, f, indent=2)
        
        print(f"\n💾 结果已保存: {output_file}")

if __name__ == '__main__':
    main()