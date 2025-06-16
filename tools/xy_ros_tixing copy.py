#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int16MultiArray, MultiArrayDimension, Float64MultiArray
from threading import Lock

class KalmanFilter:
    def __init__(self, center_x=320):
        self.center_x = center_x
        self.x = np.array([[center_x], [0.]])
        self.P = np.eye(2) * 50  # 减少初始不确定性
        self.F = np.array([[1., 0.8], [0., 0.9]])  # 减少速度影响
        self.H = np.array([[1., 0.]])
        self.R = np.array([[8.]])  # 增加测量噪声容忍度
        self.Q = np.array([[1.0, 0.], [0., 0.3]])  # 减少过程噪声
        self.initialized = False
        self.center_weight = 0.15  # 增加中心拉力
    
    def update(self, measurement):
        if not self.initialized:
            self.x[0, 0] = measurement * 0.6 + self.center_x * 0.4  # 更保守的初始化
            self.initialized = True
            return self.x[0, 0]
        
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        predicted = self.x[0, 0] * (1 - self.center_weight) + self.center_x * self.center_weight
        
        # 更严格的异常值处理
        if abs(measurement - predicted) > 25:
            measurement = predicted * 0.8 + measurement * 0.2
        
        y = measurement - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = (np.eye(2) - np.dot(K, self.H)) @ self.P
        
        return self.x[0, 0]

class TrapezoidFusionReceiver:
    def __init__(self):
        rospy.init_node('trapezoid_fusion_receiver', anonymous=True)
        
        self.bridge = CvBridge()
        self.data_lock = Lock()
        
        # 数据存储
        self.img_trapezoid = None
        self.lidar_trapezoid = None
        self.current_image = None
        self.fused_trapezoid = None
        
        # 时间戳记录
        self.img_trapezoid_time = rospy.Time(0)
        self.lidar_trapezoid_time = rospy.Time(0)
        
        # 卡尔曼滤波器
        self.image_width = 640
        self.image_height = 480
        center_x = self.image_width // 2
        self.bottom_kf = KalmanFilter(center_x)
        self.top_kf = KalmanFilter(center_x)
        
        # 动态权重参数
        self.img_history = []
        self.lidar_history = []
        self.weight_history = []  # 权重历史
        self.max_history = 10  # 增加历史长度
        
        # 稳定性参数
        self.prev_weights = (0.6, 0.4)  # 上一次权重
        self.weight_smooth_factor = 0.1  # 大幅减少权重平滑因子
        
        # 融合结果历史 - 新增
        self.fused_history = []
        self.fused_smooth_factor = 0.3  # 融合结果平滑因子
        
        # 稳定性奖励参数 - 新增
        self.stability_threshold = 5.0  # 稳定性阈值（像素）
        self.stability_reward_frames = 60  # 2秒 * 30帧/秒
        self.img_stable_count = 0
        self.lidar_stable_count = 0
        self.max_stability_weight = 0.9  # 最大稳定性奖励权重
        
        # 订阅器
        self.img_trapezoid_sub = rospy.Subscriber("/img_da_tixing", Int16MultiArray, self.img_trapezoid_callback, queue_size=1)
        self.lidar_trapezoid_sub = rospy.Subscriber("/lidar_da_tixing", Float64MultiArray, self.lidar_trapezoid_callback, queue_size=1)
        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        
        # 发布器
        self.fused_result_pub = rospy.Publisher("/fused_path_visualization/compressed", CompressedImage, queue_size=1)
        self.fused_trapezoid_pub = rospy.Publisher("/fused_da_tixing", Int16MultiArray, queue_size=1)
        
        rospy.loginfo("Trapezoid Fusion Receiver initialized")
    
    def img_trapezoid_callback(self, msg):
        try:
            with self.data_lock:
                trapezoid_data = np.array(msg.data, dtype=np.int16)
                if len(trapezoid_data) != 8:
                    return
                self.img_trapezoid = trapezoid_data.reshape(4, 2).astype(np.int32)
                self.img_trapezoid_time = rospy.Time.now()
            self.try_fusion()
        except Exception as e:
            rospy.logwarn(f"Error processing img trapezoid: {e}")
    
    def lidar_trapezoid_callback(self, msg):
        try:
            with self.data_lock:
                trapezoid_data = np.array(msg.data, dtype=np.float64)
                
                if len(trapezoid_data) == 12:
                    corners_2d = []
                    for i in range(0, 12, 3):
                        corners_2d.extend([trapezoid_data[i], trapezoid_data[i+1]])
                    self.transform_lidar_to_pixels(corners_2d)
                elif len(trapezoid_data) == 8:
                    self.lidar_trapezoid = trapezoid_data.reshape(4, 2).astype(np.int32)
                else:
                    return
                
                self.lidar_trapezoid_time = rospy.Time.now()
            self.try_fusion()
        except Exception as e:
            rospy.logwarn(f"Error processing lidar trapezoid: {e}")
    
    def transform_lidar_to_pixels(self, corners_2d):
        try:
            pc_pts = np.array([[-1, -5.14], [-1, -7.95], [2, -5.14], [2, -7.95]], dtype=np.float32)
            pixel_pts = np.array([[250, 650], [250, 609], [900, 650], [900, 601]], dtype=np.float32)
            h_matrix = cv2.getPerspectiveTransform(pc_pts, pixel_pts)
            
            lidar_points = np.array(corners_2d).reshape(4, 2).astype(np.float32)
            pixel_corners = []
            for pt in lidar_points:
                pixel_pt = cv2.perspectiveTransform(pt.reshape(1, 1, 2), h_matrix)
                pixel_corners.extend([int(pixel_pt[0, 0, 0]), int(pixel_pt[0, 0, 1])])
            
            lidar_trapezoid_original = np.array(pixel_corners).reshape(4, 2).astype(np.int32)
            self.lidar_trapezoid = np.array([
                lidar_trapezoid_original[1],  # bottom_left
                lidar_trapezoid_original[3],  # bottom_right
                lidar_trapezoid_original[2],  # top_right
                lidar_trapezoid_original[0]   # top_left
            ])
            
            # 坐标裁剪
            self.lidar_trapezoid[:, 0] = np.clip(self.lidar_trapezoid[:, 0], 0, self.image_width)
            self.lidar_trapezoid[:, 1] = np.clip(self.lidar_trapezoid[:, 1], 0, self.image_height)
            
            return True
        except Exception as e:
            rospy.logwarn(f"Error transforming lidar coordinates: {e}")
            return False
    
    def image_callback(self, msg):
        try:
            with self.data_lock:
                self.current_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'passthrough')
                if self.current_image is not None:
                    self.image_height, self.image_width = self.current_image.shape[:2]
        except Exception as e:
            rospy.logwarn(f"Error processing image: {e}")
    
    def calculate_area(self, trapezoid):
        """计算梯形面积"""
        try:
            x, y = trapezoid[:, 0], trapezoid[:, 1]
            return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
        except:
            return 0
    
    def check_stability_reward(self, current_img, current_lidar):
        """检查稳定性奖励 - 2秒内保持稳定的传感器获得更高权重"""
        # 检查图像稳定性
        if len(self.img_history) >= 2:
            recent_img_change = self.calculate_change(current_img, self.img_history[-2:])
            if recent_img_change < self.stability_threshold:
                self.img_stable_count += 1
            else:
                self.img_stable_count = 0
        else:
            self.img_stable_count = 0
        
        # 检查LiDAR稳定性
        if len(self.lidar_history) >= 2:
            recent_lidar_change = self.calculate_change(current_lidar, self.lidar_history[-2:])
            if recent_lidar_change < self.stability_threshold:
                self.lidar_stable_count += 1
            else:
                self.lidar_stable_count = 0
        else:
            self.lidar_stable_count = 0
        
        # 计算稳定性奖励权重
        img_stable = self.img_stable_count >= self.stability_reward_frames
        lidar_stable = self.lidar_stable_count >= self.stability_reward_frames
        
        # 如果只有一个稳定，给予高权重
        if img_stable and not lidar_stable:
            return self.max_stability_weight, 1.0 - self.max_stability_weight, "IMG_STABLE"
        elif lidar_stable and not img_stable:
            return 1.0 - self.max_stability_weight, self.max_stability_weight, "LIDAR_STABLE"
        elif img_stable and lidar_stable:
            # 两个都稳定时，图像6:4
            return 0.6, 0.4, "BOTH_STABLE"
        else:
            # 都不稳定，返回None表示使用动态权重
            return None, None, "DYNAMIC"
        """计算变化幅度 - 更稳定的版本"""
        if len(history) < 5:  # 需要更多历史数据才开始计算
            return 0.5  # 返回中等稳定值
        try:
            # 使用移动平均来减少噪声
            recent_history = history[-5:]
            changes = []
            
            for hist in recent_history:
                change = np.mean(np.sqrt(np.sum((current - hist) ** 2, axis=1)))
                changes.append(change)
            
            # 使用中位数而不是平均值，更抗噪声
            median_change = np.median(changes)
            return min(median_change, 30.0)  # 进一步限制最大变化值
        except:
            return 0.5
    
    def update_history(self, img_trap, lidar_trap):
        """更新历史数据"""
        if img_trap is not None:
            self.img_history.append(img_trap.copy())
            if len(self.img_history) > self.max_history:
                self.img_history.pop(0)
        
        if lidar_trap is not None:
            self.lidar_history.append(lidar_trap.copy())
            if len(self.lidar_history) > self.max_history:
                self.lidar_history.pop(0)
    
    def weighted_average_trapezoid(self, trap1, trap2):
        """超稳定的动态加权平均融合 + 稳定性奖励"""
        try:
            # 更新历史
            self.update_history(trap1, trap2)
            
            # 检查稳定性奖励
            reward_w1, reward_w2, reward_mode = self.check_stability_reward(trap1, trap2)
            
            if reward_w1 is not None:
                # 使用稳定性奖励权重
                rospy.loginfo(f"Stability Reward Mode: {reward_mode} - IMG: {reward_w1:.3f}, LiDAR: {reward_w2:.3f}")
                rospy.loginfo(f"Stability Count - IMG: {self.img_stable_count}, LiDAR: {self.lidar_stable_count}")
                
                # 更新权重历史（用于平滑过渡）
                self.prev_weights = (reward_w1, reward_w2)
                
                # 基础融合
                fused = reward_w1 * trap1 + reward_w2 * trap2
                
                # 应用融合结果的历史平滑
                final_fused = self.apply_fusion_smoothing(fused)
                return final_fused
            
            # 如果历史数据不足，使用固定权重
            if len(self.img_history) < 3 or len(self.lidar_history) < 3:
                rospy.loginfo("Using fixed weights due to insufficient history")
                fused = 0.6 * trap1 + 0.4 * trap2
                return self.apply_fusion_smoothing(fused)
            
            # 计算面积权重 - 进一步减少影响
            area1, area2 = self.calculate_area(trap1), self.calculate_area(trap2)
            total_area = area1 + area2
            if total_area == 0:
                area_w1, area_w2 = 0.5, 0.5
            else:
                area_w1, area_w2 = area1 / total_area, area2 / total_area
            
            # 计算稳定性权重 - 更保守
            change1 = self.calculate_change(trap1, self.img_history)
            change2 = self.calculate_change(trap2, self.lidar_history)
            
            # 使用更平缓的稳定性函数
            stab_w1 = 1.0 / (1.0 + change1 * 0.05)  # 进一步减少变化敏感度
            stab_w2 = 1.0 / (1.0 + change2 * 0.05)
            total_stab = stab_w1 + stab_w2
            
            if total_stab > 0:
                stab_w1, stab_w2 = stab_w1 / total_stab, stab_w2 / total_stab
            else:
                stab_w1, stab_w2 = 0.5, 0.5
            
            # 综合权重 (10%面积 + 90%稳定性) - 几乎完全依赖稳定性
            w1 = 0.1 * area_w1 + 0.9 * stab_w1
            w2 = 0.1 * area_w2 + 0.9 * stab_w2
            total_w = w1 + w2
            
            if total_w > 0:
                w1, w2 = w1 / total_w, w2 / total_w
            else:
                w1, w2 = 0.5, 0.5
            
            # 权重平滑 - 极强的历史依赖
            smooth_w1 = self.prev_weights[0] * (1 - self.weight_smooth_factor) + w1 * self.weight_smooth_factor
            smooth_w2 = 1.0 - smooth_w1
            
            # 非常严格的权重限制
            smooth_w1 = np.clip(smooth_w1, 0.3, 0.7)  # 进一步缩小范围
            smooth_w2 = 1.0 - smooth_w1
            
            # 更新权重历史
            self.prev_weights = (smooth_w1, smooth_w2)
            
            # 基础融合
            fused = smooth_w1 * trap1 + smooth_w2 * trap2
            
            # 应用融合结果的历史平滑
            final_fused = self.apply_fusion_smoothing(fused)
            
            rospy.loginfo(f"Dynamic Weights - IMG: {smooth_w1:.3f}, LiDAR: {smooth_w2:.3f} | Changes: {change1:.2f}, {change2:.2f}")
            return final_fused
            
        except Exception as e:
            rospy.logwarn(f"Error in weighted fusion: {e}")
            # 使用最保守的备用方案
            if hasattr(self, 'fused_history') and len(self.fused_history) > 0:
                return self.fused_history[-1].copy()
            return ((trap1 + trap2) / 2).astype(np.int32)
    
    def apply_fusion_smoothing(self, current_fused):
        """对融合结果进行历史平滑"""
        try:
            if len(self.fused_history) == 0:
                # 第一次融合
                self.fused_history.append(current_fused.copy())
                return current_fused.astype(np.int32)
            
            # 与历史融合结果进行平滑
            prev_fused = self.fused_history[-1]
            
            # 检查变化是否过大
            max_change = np.max(np.sqrt(np.sum((current_fused - prev_fused) ** 2, axis=1)))
            
            if max_change > 20:  # 如果变化过大，进行更强的平滑
                smoothed = prev_fused * 0.8 + current_fused * 0.2
            else:
                smoothed = prev_fused * (1 - self.fused_smooth_factor) + current_fused * self.fused_smooth_factor
            
            # 应用梯度约束 - 确保相邻顶点不会剧烈变化
            for i in range(4):
                max_point_change = 15  # 单个顶点最大变化
                point_diff = smoothed[i] - prev_fused[i]
                if np.linalg.norm(point_diff) > max_point_change:
                    point_diff = point_diff / np.linalg.norm(point_diff) * max_point_change
                    smoothed[i] = prev_fused[i] + point_diff
            
            # 更新历史
            self.fused_history.append(smoothed.copy())
            if len(self.fused_history) > 5:  # 只保留最近5个结果
                self.fused_history.pop(0)
            
            return smoothed.astype(np.int32)
            
        except Exception as e:
            rospy.logwarn(f"Error in fusion smoothing: {e}")
            return current_fused.astype(np.int32)
    
    def try_fusion(self):
        with self.data_lock:
            current_time = rospy.Time.now()
            
            img_valid = (self.img_trapezoid is not None and 
                        (current_time - self.img_trapezoid_time).to_sec() < 1.0)
            lidar_valid = (self.lidar_trapezoid is not None and 
                          (current_time - self.lidar_trapezoid_time).to_sec() < 1.0)
            
            if img_valid and lidar_valid:
                self.fused_trapezoid = self.weighted_average_trapezoid(self.img_trapezoid, self.lidar_trapezoid)
            elif img_valid:
                self.fused_trapezoid = self.img_trapezoid.copy().astype(np.int32)
            elif lidar_valid:
                self.fused_trapezoid = self.lidar_trapezoid.copy().astype(np.int32)
            
            if self.fused_trapezoid is not None:
                self.publish_fused_trapezoid()
                if self.current_image is not None:
                    self.visualize_and_publish()
    
    def publish_fused_trapezoid(self):
        try:
            if self.fused_trapezoid is not None:
                msg = Int16MultiArray()
                dim1 = MultiArrayDimension()
                dim1.label, dim1.size, dim1.stride = "points", 4, 8
                dim2 = MultiArrayDimension()
                dim2.label, dim2.size, dim2.stride = "coordinates", 2, 2
                msg.layout.dim = [dim1, dim2]
                msg.layout.data_offset = 0
                msg.data = self.fused_trapezoid.flatten().astype(np.int16).tolist()
                self.fused_trapezoid_pub.publish(msg)
        except Exception as e:
            rospy.logwarn(f"Error publishing fused trapezoid: {e}")
    
    def draw_path_rectangles(self, image, trapezoid_points):
        h, w = image.shape[:2]
        center_x = w // 2
        
        bottom_center_x = (trapezoid_points[0][0] + trapezoid_points[1][0]) // 2
        top_center_x = (trapezoid_points[2][0] + trapezoid_points[3][0]) // 2
        bottom_y, top_y = trapezoid_points[0][1], trapezoid_points[2][1]
        
        # 更强的中心拉力
        center_weight = 0.3
        bottom_center_x = bottom_center_x * (1 - center_weight) + center_x * center_weight
        top_center_x = top_center_x * (1 - center_weight) + center_x * center_weight
        
        # 不使用卡尔曼滤波，直接使用计算结果避免额外抖动
        # filtered_bottom_x = self.bottom_kf.update(bottom_center_x)
        # filtered_top_x = self.top_kf.update(top_center_x)
        filtered_bottom_x = bottom_center_x
        filtered_top_x = top_center_x
        
        # 更严格的边界限制
        max_deviation = w * 0.15
        filtered_bottom_x = np.clip(filtered_bottom_x, center_x - max_deviation, center_x + max_deviation)
        filtered_top_x = np.clip(filtered_top_x, center_x - max_deviation, center_x + max_deviation)
        
        bottom_width = abs(trapezoid_points[1][0] - trapezoid_points[0][0])
        path_height = bottom_y - top_y
        
        if path_height <= 0:
            return image
        
        overlay = image.copy()
        num_rects = min(50, path_height // 8)
        
        for i in range(num_rects):
            y_pos = bottom_y - i * 8
            if y_pos <= top_y:
                break
            
            progress = (i * 8) / path_height
            width_ratio = max(0.3, 1 - progress * 0.7)
            rect_width = int(bottom_width * width_ratio)
            
            center_x_pos = filtered_bottom_x + (filtered_top_x - filtered_bottom_x) * progress
            
            # 极强的平滑
            if i > 0:
                prev_center = filtered_bottom_x + (filtered_top_x - filtered_bottom_x) * ((i-1) * 8) / path_height
                center_x_pos = center_x_pos * 0.3 + prev_center * 0.7  # 更依赖历史
            
            center_x_pos = int(center_x_pos)
            x1, x2 = center_x_pos - rect_width // 2, center_x_pos + rect_width // 2
            y1, y2 = y_pos - 4, y_pos + 4
            
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
        
        return cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
    
    def visualize_and_publish(self):
        try:
            if self.current_image is None:
                return
            
            result = self.current_image.copy()
            current_time = rospy.Time.now()
            
            img_valid = (self.img_trapezoid is not None and 
                        (current_time - self.img_trapezoid_time).to_sec() < 1.0)
            lidar_valid = (self.lidar_trapezoid is not None and 
                          (current_time - self.lidar_trapezoid_time).to_sec() < 1.0)
            fused_valid = self.fused_trapezoid is not None
            
            # 绘制梯形
            if img_valid and self.img_trapezoid is not None:
                pts = self.img_trapezoid.reshape((-1, 1, 2))
                cv2.polylines(result, [pts], True, (0, 255, 0), 2)  # 绿色
                cv2.putText(result, "IMG", tuple(self.img_trapezoid[0] + [5, -5]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if lidar_valid and self.lidar_trapezoid is not None:
                pts = self.lidar_trapezoid.reshape((-1, 1, 2))
                cv2.polylines(result, [pts], True, (255, 0, 0), 2)  # 蓝色
                cv2.putText(result, "LIDAR", tuple(self.lidar_trapezoid[0] + [5, 15]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if fused_valid:
                pts = self.fused_trapezoid.reshape((-1, 1, 2))
                cv2.polylines(result, [pts], True, (255, 0, 255), 3)  # 紫色
                cv2.putText(result, "FUSED", tuple(self.fused_trapezoid[0] + [5, -25]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                result = self.draw_path_rectangles(result, self.fused_trapezoid)
            
            # 状态信息
            status = " | ".join([
                "IMG: OK" if img_valid else "IMG: --",
                "LIDAR: OK" if lidar_valid else "LIDAR: --",
                "FUSED: OK" if fused_valid else "FUSED: --"
            ])
            cv2.putText(result, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 发布
            msg = self.bridge.cv2_to_compressed_imgmsg(result, dst_format='jpg')
            msg.header.stamp = rospy.Time.now()
            self.fused_result_pub.publish(msg)
            
        except Exception as e:
            rospy.logwarn(f"Error in visualization: {e}")
    
    def run(self):
        rospy.loginfo("Trapezoid Fusion Receiver started")
        rospy.spin()

if __name__ == '__main__':
    try:
        receiver = TrapezoidFusionReceiver()
        receiver.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Trapezoid Fusion Receiver shutting down")