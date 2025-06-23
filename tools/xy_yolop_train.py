import argparse
import os, sys
import math
import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from tensorboardX import SummaryWriter

# 导入自定义模块
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.utils import DataLoaderX
import lib.dataset as dataset
from lib.config import cfg_xy as cfg
from lib.config import update_config_xy as update_config
from lib.core.loss import get_loss
from lib.core.function import train_xy, validate
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer, save_checkpoint, create_logger, select_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Multitask network')
    
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='runs/')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    
    return parser.parse_args()


def setup_distributed_training():
    """设置分布式训练环境"""
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    global_rank = int(os.environ.get('RANK', -1))
    return world_size, global_rank


def load_model_weights(model, logger, optimizer):
    """加载预训练模型权重"""
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    last_epoch = -1
    
    # 加载完整预训练模型
    if os.path.exists(cfg.MODEL.PRETRAINED):
        logger.info(f"=> loading model '{cfg.MODEL.PRETRAINED}'")
        checkpoint = torch.load(cfg.MODEL.PRETRAINED)
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> loaded checkpoint '{cfg.MODEL.PRETRAINED}' (epoch {checkpoint['epoch']})")
    
    # 加载检测分支权重
    if os.path.exists(cfg.MODEL.PRETRAINED_DET):
        logger.info(f"=> loading det branch weights from '{cfg.MODEL.PRETRAINED_DET}'")
        det_idx_range = [str(i) for i in range(0, 25)]
        model_dict = model.state_dict()
        checkpoint = torch.load(cfg.MODEL.PRETRAINED_DET)
        checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                          if k.split(".")[1] in det_idx_range}
        model_dict.update(checkpoint_dict)
        model.load_state_dict(model_dict)
        logger.info("=> loaded det branch checkpoint successfully")
    
    # 自动恢复训练
    checkpoint_file = os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET, 'checkpoint.pth')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info(f"=> loading checkpoint '{checkpoint_file}'")
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})")
    
    return begin_epoch, last_epoch


def create_data_loaders(rank):
    """创建训练和验证数据加载器"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 训练数据集
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg, is_train=True, inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([transforms.ToTensor(), normalize])
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None
    
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE and rank == -1),
        num_workers=cfg.WORKERS,
        sampler=train_sampler,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    
    # 验证数据集 (仅主进程需要)
    valid_loader = None
    valid_dataset = None
    if rank in [-1, 0]:
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg, is_train=False, inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([transforms.ToTensor(), normalize])
        )
        
        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )
    
    return train_loader, valid_loader, valid_dataset


def should_validate_and_save(epoch):
    """判断是否应该进行验证和保存模型"""
    return (epoch % cfg.TRAIN.VAL_FREQ == 0 or 
            epoch == cfg.TRAIN.END_EPOCH or 
            epoch in list(range(181, 200)) or 
            epoch in [162, 165, 167, 170, 172, 175, 178])


def main():
    # 解析参数和配置
    args = parse_args()
    update_config(cfg, args)
    
    # 设置分布式训练
    world_size, rank = setup_distributed_training()
    
    # 创建日志和输出目录
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.LOG_DIR, 'train', rank=rank)
    
    # 设置writer (仅主进程)
    writer_dict = None
    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))
        logger.info(cfg)
        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    
    # CUDNN设置
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    
    # 设备选择和分布式初始化
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) if not cfg.DEBUG else select_device(logger, 'cpu')
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    
    # 构建模型
    print("Building model...")
    model = get_net(cfg).to(device)
    criterion = get_loss(cfg, device, model)
    optimizer = get_optimizer(cfg, model)
    
    # 加载模型权重
    begin_epoch, last_epoch = load_model_weights(model, logger, optimizer) if rank in [-1, 0] else (cfg.TRAIN.BEGIN_EPOCH, -1)
    
    # 设置并行训练
    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
    elif rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    # 模型参数设置
    model.gr = 1.0
    model.nc = 1
    
    # 创建数据加载器
    print("Loading data...")
    train_loader, valid_loader, valid_dataset = create_data_loaders(rank)
    
    # 学习率调度器
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # 训练设置
    num_batch = len(train_loader)
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    print('=> Start training...')
    
    # 主训练循环
    for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        
        # 训练一个epoch
        train_xy(cfg, train_loader, model, criterion, optimizer, scaler,
                epoch, num_batch, num_warmup, writer_dict, logger, device, rank)
        
        lr_scheduler.step()
        
        # 验证和保存 (仅主进程)
        if should_validate_and_save(epoch) and rank in [-1, 0]:
            # 验证
            da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict, logger, device, rank
            )
            
            # 打印结果
            msg = f'Epoch: [{epoch}] Loss({total_loss:.3f})\n' \
                  f'Driving area Segment: Acc({da_segment_results[0]:.3f}) IOU({da_segment_results[1]:.3f}) mIOU({da_segment_results[2]:.3f})\n' \
                  f'Lane line Segment: Acc({ll_segment_results[0]:.3f}) IOU({ll_segment_results[1]:.3f}) mIOU({ll_segment_results[2]:.3f})\n' \
                  f'Detect: P({detect_results[0]:.3f}) R({detect_results[1]:.3f}) mAP@0.5({detect_results[2]:.3f}) mAP@0.5:0.95({detect_results[3]:.3f})\n' \
                  f'Time: inference({times[0]:.4f}s/frame) nms({times[1]:.4f}s/frame)'
            logger.info(msg)
            
            # 保存检查点
            save_checkpoint(
                epoch=epoch, name=cfg.MODEL.NAME, model=model, optimizer=optimizer,
                output_dir=final_output_dir, filename=f'epoch-{epoch}.pth'
            )
            save_checkpoint(
                epoch=epoch, name=cfg.MODEL.NAME, model=model, optimizer=optimizer,
                output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), filename='checkpoint.pth'
            )
    
    # 保存最终模型
    if rank in [-1, 0]:
        final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info(f'=> Saving final model state to {final_model_state_file}')
        model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
        torch.save(model_state, final_model_state_file)
        writer_dict['writer'].close()
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()