import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path

from lib.core.evaluate import ConfusionMatrix, SegmentationMetric
from lib.core.general import non_max_suppression, check_img_size, box_iou, ap_per_class, box_iou, ap_per_class
from lib.utils.utils import time_synchronized, xywh2xyxy, scale_coords, clip_coords, _coco80_to_coco91_class

class Validator:
    def __init__(self, config, val_loader, val_dataset, model, criterion, output_dir, console_logger_instance, wandb_logger_instance):
        self.config = config
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.model = model
        self.criterion = criterion
        self.output_dir = output_dir
        self.console_logger_instance = console_logger_instance
        self.wandb_logger_instance = wandb_logger_instance
        self.device = next(model.parameters()).device

    def validate(self, epoch):
        logger = self.console_logger_instance.get_logger()
        max_stride = 32
        _, imgsz = [check_img_size(x, s=max_stride) for x in self.config.MODEL.IMAGE_SIZE]
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)

        detection_classes = getattr(self.model, 'nc', 1)
        da_seg_classes = getattr(self.config, 'num_seg_class', 2)
        ll_seg_classes = 2

        confusion_matrix = ConfusionMatrix(nc=detection_classes)
        da_metric = SegmentationMetric(da_seg_classes)
        ll_metric = SegmentationMetric(ll_seg_classes)

        losses, da_acc_seg, da_IoU_seg, da_mIoU_seg, ll_acc_seg, ll_IoU_seg, ll_mIoU_seg, T_inf, T_nms = [AverageMeter() for _ in range(9)]

        self.model.eval()
        stats = []
        jdict = []
        seen = 0
        names = {k: v for k, v in enumerate(self.model.names if hasattr(self.model, 'names') else self.model.module.names)}
        coco91class = _coco80_to_coco91_class()

        with torch.no_grad():
            for batch_i, (img, target, paths, shapes) in tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc='Validation'):
                img = img.to(self.device, non_blocking=True)
                target = [t.to(self.device) for t in target]
                nb, _, height, width = img.shape

                pad_h, pad_w = self._get_padding(shapes)

                t = time_synchronized()
                det_out, da_seg_out, ll_seg_out = self.model(img)
                t_inf = time_synchronized() - t
                if batch_i > 0: T_inf.update(t_inf / img.size(0), img.size(0))

                inf_out, train_out = det_out

                if da_seg_out is not None and target[1] is not None:
                    _, da_predict = torch.max(da_seg_out, 1)
                    _, da_gt = torch.max(target[1], 1)
                    da_predict_crop, da_gt_crop = self._crop_padding(da_predict, da_gt, height, width, pad_h, pad_w)
                    da_metric.reset()
                    da_metric.addBatch(da_predict_crop.cpu(), da_gt_crop.cpu())
                    da_acc_seg.update(da_metric.pixelAccuracy(), img.size(0))
                    da_IoU_seg.update(da_metric.IntersectionOverUnion(), img.size(0))
                    da_mIoU_seg.update(da_metric.meanIntersectionOverUnion(), img.size(0))

                if ll_seg_out is not None and target[2] is not None:
                    _, ll_predict = torch.max(ll_seg_out, 1)
                    _, ll_gt = torch.max(target[2], 1)
                    ll_predict_crop, ll_gt_crop = self._crop_padding(ll_predict, ll_gt, height, width, pad_h, pad_w)
                    ll_metric.reset()
                    ll_metric.addBatch(ll_predict_crop.cpu(), ll_gt_crop.cpu())
                    ll_acc_seg.update(ll_metric.lineAccuracy(), img.size(0))
                    ll_IoU_seg.update(ll_metric.IntersectionOverUnion(), img.size(0))
                    ll_mIoU_seg.update(ll_metric.meanIntersectionOverUnion(), img.size(0))

                total_loss, _ = self.criterion((train_out, da_seg_out, ll_seg_out), target, shapes, self.model, img)
                losses.update(total_loss.item(), img.size(0))

                t = time_synchronized()
                output = non_max_suppression(inf_out, conf_thres=self.config.TEST.NMS_CONF_THRESHOLD, iou_thres=self.config.TEST.NMS_IOU_THRESHOLD)
                t_nms = time_synchronized() - t
                if batch_i > 0: T_nms.update(t_nms / img.size(0), img.size(0))

                for si, pred in enumerate(output):
                    path = Path(paths[si])
                    seen += 1
                    nlabel = (target[0][si].sum(dim=1) > 0).sum()
                    tcls = target[0][si, :nlabel, 0].tolist() if nlabel else []

                    if len(pred) == 0:
                        if nlabel: stats.append((torch.zeros(0, iouv.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                        continue

                    predn = pred.clone()
                    scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

                    tbox = xywh2xyxy(target[0][si, :nlabel, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])

                    gt_boxes_for_log = [[*box.tolist(), cls] for box, cls in zip(tbox, tcls)]
                    self.wandb_logger_instance.cache_valid_detection_image(img[si], pred.tolist(), gt_boxes_for_log, names, epoch, path.name)

                    if da_seg_out is not None:
                        self.wandb_logger_instance.cache_valid_seg_image(img[si], da_predict_crop[si].cpu().numpy(), da_gt_crop[si].cpu().numpy(), 'da_seg', epoch, path.name)

                    if ll_seg_out is not None:
                        self.wandb_logger_instance.cache_valid_seg_image(img[si], ll_predict_crop[si].cpu().numpy(), ll_gt_crop[si].cpu().numpy(), 'll_seg', epoch, path.name)

                    correct = self._compute_metrics(predn, target[0][si, :nlabel, :], tbox, iouv)
                    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        mp, mr, map50, map = self._process_stats(stats, names)
        self.model.float()

        self._log_results(epoch, losses, mp, mr, map50, map, da_acc_seg, da_IoU_seg, da_mIoU_seg, ll_acc_seg, ll_IoU_seg, ll_mIoU_seg, T_inf, T_nms)

        da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
        ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)
        detect_result = np.asarray([mp, mr, map50, map])
        t = [T_inf.avg, T_nms.avg]

        return da_segment_result, ll_segment_result, detect_result, losses.avg, None, t

    def _get_padding(self, shapes):
        if len(shapes) > 0 and len(shapes[0]) > 1 and isinstance(shapes[0][1], (tuple, list)) and len(shapes[0][1]) == 2:
            pad_info = shapes[0][1]
            if isinstance(pad_info[0], (int, float)) and isinstance(pad_info[1], (int, float)):
                return int(pad_info[0]), int(pad_info[1])
        return 0, 0

    def _crop_padding(self, pred, gt, h, w, pad_h, pad_w):
        if h > 2 * pad_h and w > 2 * pad_w:
            return pred[:, pad_h:h - pad_h, pad_w:w - pad_w], gt[:, pad_h:h - pad_h, pad_w:w - pad_w]
        return pred, gt

    def _compute_metrics(self, pred, labels, tbox, iouv):
        correct = torch.zeros(pred.shape[0], iouv.numel(), dtype=torch.bool, device=self.device)
        if labels.shape[0]:
            iou = box_iou(pred[:, :4], tbox)
            x = torch.where(iou > iouv[0])
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                matches = torch.from_numpy(matches).to(self.device)
                correct[matches[:, 0].long()] = matches[:, 2:3] > iouv
        return correct

    def _process_stats(self, stats, names):
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)
            return p.mean(), r.mean(), ap50.mean(), ap.mean()
        return 0.0, 0.0, 0.0, 0.0

    def _log_results(self, epoch, losses, mp, mr, map50, map, da_acc, da_iou, da_miou, ll_acc, ll_iou, ll_miou, t_inf, t_nms):
        logger = self.console_logger_instance.get_logger()
        logger.info(f'Validation Results - Epoch {epoch}')
        logger.info(f'Overall Loss: {losses.avg:.6f}')
        logger.info(f'Detection: P={mp:.4f}, R={mr:.4f}, mAP@.5={map50:.4f}, mAP@.5:.95={map:.4f}')
        logger.info(f'DA Seg: Acc={da_acc.avg:.4f}, IoU={da_iou.avg:.4f}, mIoU={da_miou.avg:.4f}')
        logger.info(f'LL Seg: Acc={ll_acc.avg:.4f}, IoU={ll_iou.avg:.4f}, mIoU={ll_miou.avg:.4f}')
        logger.info(f'Speed: Inf={t_inf.avg:.4f}s, NMS={t_nms.avg:.4f}s')
        self.wandb_logger_instance.log_validation_images(epoch)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
