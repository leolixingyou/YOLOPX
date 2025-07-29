#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class YOLOX_Loss(nn.Module):
    def __init__(self, num_classes, strides, in_channels):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.in_channels = in_channels
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        
        # Initialize expanded_strides - will be set when processing predictions
        self.expanded_strides = None

    def forward(self, predictions, labels, imgs):
        return self.get_losses(imgs.shape[2:], predictions, labels)
    
    def get_grid(self, grid_size, stride):
        """Generate grid coordinates for a given grid size and stride."""
        grid_h, grid_w = grid_size
        device = next(self.parameters()).device
        
        grid_y, grid_x = torch.meshgrid([torch.arange(grid_h, device=device), 
                                        torch.arange(grid_w, device=device)])
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2).float()
        grid = grid * stride
        
        return grid

    def get_losses(self, imgs_shape, predictions, labels):
        bbox_preds = []
        obj_preds = []
        cls_preds = []
        grids = []
        expanded_strides = []

        for i, pred in enumerate(predictions):
            stride = self.strides[i]
            grid = self.get_grid(pred.shape[2:], stride)
            grids.append(grid)
            
            # Create expanded strides for this level
            grid_h, grid_w = pred.shape[2:]
            expanded_stride = torch.full((1, grid_h * grid_w, 1), stride, 
                                       dtype=torch.float32, device=pred.device)
            expanded_strides.append(expanded_stride)
            
            bbox_pred = pred[:, :4, :, :]
            obj_pred = pred[:, 4:5, :, :]
            cls_pred = pred[:, 5:, :, :]

            bbox_preds.append(bbox_pred.flatten(start_dim=2).permute(0, 2, 1))
            obj_preds.append(obj_pred.flatten(start_dim=2).permute(0, 2, 1))
            cls_preds.append(cls_pred.flatten(start_dim=2).permute(0, 2, 1))

        bbox_preds = torch.cat(bbox_preds, dim=1)
        obj_preds = torch.cat(obj_preds, dim=1)
        cls_preds = torch.cat(cls_preds, dim=1)
        grids = torch.cat(grids, dim=1)
        
        # Set expanded_strides for use in get_assignments
        self.expanded_strides = torch.cat(expanded_strides, dim=1)

        # Handle different label shapes - if labels is 2D, it's already flattened
        if len(labels.shape) == 2:
            nlabel = torch.tensor([labels.shape[0]], device=labels.device)
        else:
            nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        
        total_num_anchors = bbox_preds.shape[1]
        
        # Ensure labels can be reshaped to [N, 6] format
        if labels.numel() % 6 != 0:
            # Pad with zeros or truncate to make it divisible by 6
            remainder = labels.numel() % 6
            if remainder != 0:
                padding = 6 - remainder
                labels = torch.cat([labels.view(-1), torch.zeros(padding, device=labels.device)])
        
        labels = labels.reshape(-1, 6)
        
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(bbox_preds.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = bbox_preds.new_zeros((0, self.num_classes))
                reg_target = bbox_preds.new_zeros((0, 4))
                l1_target = bbox_preds.new_zeros((0, 4))
                obj_target = bbox_preds.new_zeros((total_num_anchors, 1))
                fg_mask = bbox_preds.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[labels[:, 0] == batch_idx, 2:6]
                gt_classes = labels[labels[:, 0] == batch_idx, 1]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                
                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    num_gt, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, grids
                )
                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        bbox_preds.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        grids[0][fg_mask],
                        self.strides
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            l1_targets.append(l1_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
        if self.use_l1:
            loss_l1 = (self.l1_loss(bbox_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
        return loss

    def get_grid(self, grid_size, stride):
        yv, xv = torch.meshgrid([torch.arange(grid_size[0]), torch.arange(grid_size[1])])
        grid = torch.stack((xv, yv), 2).view(1, grid_size[0] * grid_size[1], 2).float()
        return grid

    @torch.no_grad()
    def get_assignments(self, num_gt, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, grids):
        num_anchors = bboxes_preds_per_image.shape[0]
        
        # fg_mask, is_in_boxes_and_center
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, grids)

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image)
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        
        cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        obj_preds_ = obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        
        pair_wise_cls_loss = F.binary_cross_entropy(
            cls_preds_.sqrt_(), gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
            reduction="none"
        ).sum(-1)
        
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center)
        
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask
        )
        del cost, pair_wise_cls_loss, pair_wise_ious_loss, cls_preds_, obj_preds_
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def get_in_boxes_info(self, gt_bboxes_per_image, grids):
        expanded_strides_per_image = self.expanded_strides[0]
        x_shifts_per_image = grids[:, :, 0]
        y_shifts_per_image = grids[:, :, 1]

        x_centers_per_image = (x_shifts_per_image + 0.5) * expanded_strides_per_image
        y_centers_per_image = (y_shifts_per_image + 0.5) * expanded_strides_per_image

        x0_gt = gt_bboxes_per_image[:, 0]
        y0_gt = gt_bboxes_per_image[:, 1]
        x1_gt = gt_bboxes_per_image[:, 2]
        y1_gt = gt_bboxes_per_image[:, 3]

        center_radius = 2.5
        b_l = x_centers_per_image - x0_gt[:, None]
        b_r = x1_gt[:, None] - x_centers_per_image
        b_t = y_centers_per_image - y0_gt[:, None]
        b_b = y1_gt[:, None] - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        center_dist = torch.sqrt(
            (x_centers_per_image - (x0_gt + x1_gt)[:, None] / 2) ** 2 +
            (y_centers_per_image - (y0_gt + y1_gt)[:, None] / 2) ** 2
        )
        gt_box_diag = torch.sqrt(
            (x1_gt - x0_gt) ** 2 + (y1_gt - y0_gt) ** 2
        )
        min_center_dist, _ = torch.min(center_dist, dim=0)
        is_in_centers = min_center_dist < (gt_box_diag * center_radius).max()

        is_in_centers_all = is_in_centers.sum(dim=0) > 0
        
        is_in_boxes_and_center = (is_in_boxes_all | is_in_centers_all)
        is_in_boxes_anchor = is_in_boxes_all | (is_in_centers_all & is_in_boxes_all)
        
        fg_mask = is_in_boxes_anchor & is_in_boxes_and_center
        return fg_mask, is_in_boxes_and_center

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )
            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i + 1e-16)

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.shape[1])
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1
        
        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
