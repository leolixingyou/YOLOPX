#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import torchvision.transforms as transforms
import time

class KalmanFilter:
    """改进的卡尔曼滤波器，增加稳定性"""
    def __init__(self, center_x=320):  # 传入图像中心x坐标
        self.center_x = center_x
        self.x = np.array([[center_x], [0.]])  # 初始化为图像中心
        self.P = np.eye(2) * 100  # 减小初始不确定性
        self.F = np.array([[1., 1.], [0., 1.]])
        self.H = np.array([[1., 0.]])
        self.R = np.array([[3.]])  # 减小测量噪声
        self.Q = np.array([[0.5, 0.], [0., 0.5]])  # 减小过程噪声
        self.initialized = False
        self.center_weight = 0.05  # 中心拉力权重
    
    def update(self, measurement):
        if not self.initialized:
            # 初始化时考虑图像中心
            self.x[0, 0] = measurement * 0.8 + self.center_x * 0.2
            self.initialized = True
            return self.x[0, 0]
        
        # Predict
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        predicted = self.x[0, 0]
        
        # 添加中心拉力
        predicted = predicted * (1 - self.center_weight) + self.center_x * self.center_weight
        
        # 更严格的测量验证
        measurement_diff = abs(measurement - predicted)
        if measurement_diff > 15:  # 如果偏差太大，减少测量权重
            measurement = predicted * 0.7 + measurement * 0.3
        
        # Update
        y = measurement - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = (np.eye(2) - np.dot(K, self.H)) @ self.P
        
        return self.x[0, 0]

class ImageProcessor:
    def __init__(self, det_weights, seg_weights, img_size=640, conf_thres=0.25):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.frame_id = 0
        
        # Detection model
        self.det_model = YOLO(det_weights)
        
        # Segmentation model
        self.seg_session = ort.InferenceSession(seg_weights)
        self.seg_input_name = self.seg_session.get_inputs()[0].name
        self.seg_outputs = [out.name for out in self.seg_session.get_outputs()]
        
        # Transforms for segmentation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        center_x = img_size // 2
        self.bottom_kf = KalmanFilter(center_x)
        self.top_kf = KalmanFilter(center_x)
        
        # 添加用于存储最新梯形数据的变量
        self.latest_trapezoid = None
    
    def preprocess_segmentation(self, image):
        """Preprocess image for segmentation"""
        h, w = image.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        
        pad_x = (self.img_size - new_w) // 2
        pad_y = (self.img_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        rgb_img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor_img = self.transform(rgb_img).unsqueeze(0)
        
        return tensor_img, {'scale': scale, 'pad': (pad_x, pad_y), 'original_shape': (h, w)}
    
    def postprocess_segmentation(self, outputs, original_img, shapes_info):
        """Postprocess segmentation outputs"""
        if len(outputs) < 6:
            return None, None
        
        da_seg = torch.from_numpy(outputs[4])
        ll_seg = torch.from_numpy(outputs[5])
        
        h, w = shapes_info['original_shape']
        pad_x, pad_y = shapes_info['pad']
        
        # Crop padding
        da_predict = da_seg[:, :, pad_y:(self.img_size-pad_y), pad_x:(self.img_size-pad_x)]
        ll_predict = ll_seg[:, :, pad_y:(self.img_size-pad_y), pad_x:(self.img_size-pad_x)]
        
        # Resize to original
        da_mask = F.interpolate(da_predict, size=(h, w), mode='bilinear')
        ll_mask = F.interpolate(ll_predict, size=(h, w), mode='bilinear')
        
        _, da_mask = torch.max(da_mask, 1)
        _, ll_mask = torch.max(ll_mask, 1)
        
        da_mask = da_mask.int().squeeze().cpu().numpy()
        ll_mask = ll_mask.int().squeeze().cpu().numpy()
        
        # Process road mask
        road_mask = np.zeros_like(da_mask)
        road_mask[(da_mask - ll_mask) == 1] = 1
        
        return road_mask, ll_mask
    
    def fit_trapezoid_from_lanes(self, ll_mask):
        """改进的梯形拟合方法，减少路径偏移"""
        h, w = ll_mask.shape
        center_x = w // 2
        center_y = h // 2
        
        # Find lane line points
        lane_points = np.where(ll_mask == 1)
        if len(lane_points[0]) == 0:
            # Default trapezoid centered on image center
            bottom_width = w * 0.6  # 减小默认宽度
            top_width = w * 0.3
            height = h * 0.6
            
            trapezoid = np.array([
                [center_x - bottom_width//2, center_y + height//2],
                [center_x + bottom_width//2, center_y + height//2],
                [center_x + top_width//2, center_y - height//2],
                [center_x - top_width//2, center_y - height//2]
            ], dtype=np.int32)
            
            # 更新最新梯形数据
            self.latest_trapezoid = trapezoid.copy()
            return trapezoid
        
        # 改进：分层检测车道线边界
        lane_y = lane_points[0]
        lane_x = lane_points[1]
        
        # 将图像分为多个水平条带进行分析
        num_strips = 10
        strip_height = h // num_strips
        
        left_points = []
        right_points = []
        center_points = []
        
        for i in range(num_strips):
            y_start = i * strip_height
            y_end = (i + 1) * strip_height
            
            # 找到当前条带内的车道线点
            strip_mask = (lane_y >= y_start) & (lane_y < y_end)
            if not np.any(strip_mask):
                continue
                
            strip_x = lane_x[strip_mask]
            strip_y = lane_y[strip_mask]
            
            if len(strip_x) > 0:
                # 寻找左右边界
                x_min = np.min(strip_x)
                x_max = np.max(strip_x)
                y_avg = np.mean(strip_y)
                
                # 检查是否有明显的左右分离
                x_sorted = np.sort(strip_x)
                if len(x_sorted) > 10:  # 有足够的点
                    # 寻找可能的间隙
                    gaps = np.diff(x_sorted)
                    max_gap_idx = np.argmax(gaps)
                    
                    if gaps[max_gap_idx] > 20:  # 如果有明显间隙
                        left_boundary = x_sorted[max_gap_idx]
                        right_boundary = x_sorted[max_gap_idx + 1]
                        center_x_strip = (left_boundary + right_boundary) / 2
                    else:
                        # 没有明显间隙，使用整体中心
                        center_x_strip = (x_min + x_max) / 2
                else:
                    center_x_strip = (x_min + x_max) / 2
                
                left_points.append([x_min, y_avg])
                right_points.append([x_max, y_avg])
                center_points.append([center_x_strip, y_avg])
        
        # 如果没有足够的点，使用默认值
        if len(center_points) < 3:
            bottom_width = w * 0.6
            top_width = w * 0.3
            height = h * 0.6
            
            trapezoid = np.array([
                [center_x - bottom_width//2, center_y + height//2],
                [center_x + bottom_width//2, center_y + height//2],
                [center_x + top_width//2, center_y - height//2],
                [center_x - top_width//2, center_y - height//2]
            ], dtype=np.int32)
            
            # 更新最新梯形数据
            self.latest_trapezoid = trapezoid.copy()
            return trapezoid
        
        # 使用多项式拟合中心线
        center_points = np.array(center_points)
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        
        # 按y坐标排序
        center_points = center_points[np.argsort(center_points[:, 1])]
        left_points = left_points[np.argsort(left_points[:, 1])]
        right_points = right_points[np.argsort(right_points[:, 1])]
        
        # 获取顶部和底部的点
        bottom_y = int(np.max(center_points[:, 1]))
        top_y = int(np.min(center_points[:, 1]))
        
        # 拟合中心线 - 使用加权方法，优先考虑图像中心
        weights = np.exp(-0.1 * np.abs(center_points[:, 0] - center_x))  # 越靠近中心权重越大
        
        if len(center_points) >= 2:
            # 线性拟合中心线
            z = np.polyfit(center_points[:, 1], center_points[:, 0], 1, w=weights)
            poly = np.poly1d(z)
            
            # 计算顶部和底部的中心点
            bottom_center_x = poly(bottom_y)
            top_center_x = poly(top_y)
            
            # 限制中心点不要偏离图像中心太远
            max_deviation = w * 0.3
            bottom_center_x = np.clip(bottom_center_x, center_x - max_deviation, center_x + max_deviation)
            top_center_x = np.clip(top_center_x, center_x - max_deviation, center_x + max_deviation)
        else:
            bottom_center_x = center_x
            top_center_x = center_x
        
        # 计算底部和顶部的宽度
        bottom_width = min(w * 0.8, np.max(right_points[:, 0]) - np.min(left_points[:, 0]))
        top_width = bottom_width * 0.5  # 顶部宽度为底部的一半
        
        # 构建梯形
        trapezoid = np.array([
            [int(bottom_center_x - bottom_width//2), bottom_y],
            [int(bottom_center_x + bottom_width//2), bottom_y],
            [int(top_center_x + top_width//2), top_y],
            [int(top_center_x - top_width//2), top_y]
        ], dtype=np.int32)
        
        # 更新最新梯形数据
        self.latest_trapezoid = trapezoid.copy()
        return trapezoid

    def draw_path_rectangles(self, image, trapezoid_points):
        """改进的路径绘制方法，增加稳定性"""
        h, w = image.shape[:2]
        center_x = w // 2
        
        # Get bottom and top center points
        bottom_center_x = (trapezoid_points[0][0] + trapezoid_points[1][0]) // 2
        top_center_x = (trapezoid_points[2][0] + trapezoid_points[3][0]) // 2
        bottom_y = trapezoid_points[0][1]
        top_y = trapezoid_points[2][1]
        
        # 添加中心拉力 - 让路径向图像中心收敛
        center_weight = 0.1  # 中心拉力权重
        bottom_center_x = bottom_center_x * (1 - center_weight) + center_x * center_weight
        top_center_x = top_center_x * (1 - center_weight) + center_x * center_weight
        
        # Apply Kalman filter to center points
        filtered_bottom_x = self.bottom_kf.update(bottom_center_x)
        filtered_top_x = self.top_kf.update(top_center_x)
        
        # 再次应用中心约束
        max_deviation = w * 0.25  # 最大偏离距离
        filtered_bottom_x = np.clip(filtered_bottom_x, center_x - max_deviation, center_x + max_deviation)
        filtered_top_x = np.clip(filtered_top_x, center_x - max_deviation, center_x + max_deviation)
        
        # Calculate trapezoid parameters
        bottom_width = abs(trapezoid_points[1][0] - trapezoid_points[0][0])
        path_height = bottom_y - top_y
        
        if path_height <= 0:
            return image
        
        # Create overlay for transparency
        overlay = image.copy()
        
        # Draw rectangles from bottom to top
        num_rects = min(50, path_height // 8)  # 稍微增加矩形密度
        
        for i in range(num_rects):
            y_pos = bottom_y - i * 8
            if y_pos <= top_y:
                break
            
            # Calculate rectangle width with smoother transition
            progress = (i * 8) / path_height if path_height > 0 else 0
            width_ratio = max(0.3, 1 - progress * 0.7)  # 更平滑的宽度变化
            rect_width = int(bottom_width * width_ratio)
            
            # Smooth interpolation between filtered points
            center_x_pos = filtered_bottom_x + (filtered_top_x - filtered_bottom_x) * progress
            
            # 添加轻微的平滑处理
            if i > 0:
                # 与前一个矩形的中心进行平滑过渡
                prev_center = filtered_bottom_x + (filtered_top_x - filtered_bottom_x) * ((i-1) * 8) / path_height
                center_x_pos = center_x_pos * 0.7 + prev_center * 0.3
            
            center_x_pos = int(center_x_pos)
            
            # Draw rectangle
            x1 = center_x_pos - rect_width // 2
            x2 = center_x_pos + rect_width // 2
            y1 = y_pos - 4
            y2 = y_pos + 4
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            
            # 使用渐变颜色 - 远处更透明
            alpha = max(0.3, 1 - progress * 0.5)
            color = (0, 255, 255)  # 黄色
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Apply transparency
        result = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
        return result
    
    def get_latest_trapezoid(self):
        """获取最新的梯形数据"""
        return self.latest_trapezoid
    
    def draw_timing(self, image, text, inference_time):
        """Draw timing information with white background and black text"""
        timing_text = f"{text}: {inference_time:.1f}ms"
        text_size = cv2.getTextSize(timing_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(image, (10, 10), (20 + text_size[0], 40 + text_size[1]), (255, 255, 255), -1)
        cv2.putText(image, timing_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    def visualize_detection(self, image, tracks, inference_time):
        """Draw detection results with tracking"""
        result = image.copy()
        
        # Draw timing information with white background and black text
        self.draw_timing(result, "Detection", inference_time)
        
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            score = track[5] if len(track) > 5 else 0
            
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result, f'ID:{track_id}', (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f'{score:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result
    
    def visualize_segmentation(self, image, da_mask, ll_mask, inference_time):
        """Draw segmentation results with path planning"""
        result = image.copy()
        
        # Draw timing information with white background and black text
        self.draw_timing(result, "Segmentation", inference_time)
        
        if da_mask is not None:
            # Road area in green
            road_overlay = np.zeros_like(image)
            road_overlay[da_mask == 1] = [0, 255, 0]

            result = cv2.addWeighted(result, 0.7, road_overlay, 0.3, 0)

            trapezoid_points = self.fit_trapezoid_from_lanes(da_mask)

            # Draw path rectangles with Kalman filtering
            result = self.draw_path_rectangles(result, trapezoid_points)
        
        if ll_mask is not None:
            # Draw lane lines in red
            result[ll_mask == 1] = [0, 0, 255]
            
            # Draw trapezoid outline for debugging (只有在有trapezoid_points时才绘制)
            if 'trapezoid_points' in locals():
                cv2.polylines(result, [trapezoid_points], True, (255, 0, 0), 2)
        
        return result
    
    def process_detection(self, image):
        """Process detection with tracking"""
        start_time = time.time()
        
        try:
            self.frame_id += 1
            
            # Use YOLO for detection and tracking
            results = self.det_model.track(image, persist=True)
            
            if not results or not results[0].boxes:
                inference_time = (time.time() - start_time) * 1000
                return self.visualize_detection(image, [], inference_time)
            
            boxes = results[0].boxes
            tracks = []
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                
                if conf > self.conf_thres:
                    tracks.append([x1, y1, x2, y2, track_id, conf])
            
            inference_time = (time.time() - start_time) * 1000
            return self.visualize_detection(image, tracks, inference_time)
            
        except Exception as e:
            print(f"Detection error: {e}")
            inference_time = (time.time() - start_time) * 1000
            return self.visualize_detection(image, [], inference_time)
    
    def process_segmentation(self, image):
        """Process segmentation"""
        start_time = time.time()
        
        try:
            # Preprocess
            tensor_img, shapes_info = self.preprocess_segmentation(image)
            img_np = tensor_img.cpu().numpy().astype(np.float32)
            
            # Inference
            ort_inputs = {self.seg_input_name: img_np}
            outputs = self.seg_session.run(self.seg_outputs, ort_inputs)
            
            # Postprocess
            da_mask, ll_mask = self.postprocess_segmentation(outputs, image, shapes_info)
            
            inference_time = (time.time() - start_time) * 1000
            return self.visualize_segmentation(image, da_mask, ll_mask, inference_time)
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            inference_time = (time.time() - start_time) * 1000
            return self.visualize_segmentation(image, None, None, inference_time)

# Main execution
if __name__ == '__main__':
    from xy_ros_comunitation import ROSImageNode
    
    det_weights = "/workspace/onnx_converter/yolov10n.pt"
    seg_weights = "/workspace/onnx_converter/yolopx.onnx"
    
    processor = ImageProcessor(det_weights, seg_weights)
    ros_node = ROSImageNode(processor)
    ros_node.run()