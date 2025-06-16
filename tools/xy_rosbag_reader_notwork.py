#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import rosbag
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import argparse
from datetime import datetime
import time
import threading

class SimpleBagProcessor:
    def __init__(self, bag_path, output_dir="/workspace/results"):
        """
        简化的bag处理器
        """
        self.bag_path = bag_path
        self.output_dir = output_dir
        self.bridge = CvBridge()
        
        # 创建输出目录
        self.create_output_directory()
        
        # 视频写入器
        self.video_writers = {}
        self.frame_counters = {}
        self.video_fps = 15
        self.video_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 输出视频配置
        self.output_videos = {
            'original': 'original_images.mp4',
            'detection': 'detection_results.mp4',
            'segmentation': 'segmentation_results.mp4',
            'lidar_perspective': 'lidar_perspective.mp4',
            'fused': 'fused_results.mp4'
        }
        
        # 数据缓存
        self.latest_image = None
        self.latest_lidar_points = None
        self.img_trapezoid = None
        self.lidar_trapezoid = None
        self.data_lock = threading.Lock()
        
        # 透视变换矩阵
        self.setup_perspective_transform()
        
        # 简单的滤波器
        self.center_x = 320
        self.filtered_bottom_x = self.center_x
        self.filtered_top_x = self.center_x
        
        print(f"SimpleBagProcessor initialized")
        print(f"Bag file: {self.bag_path}")
        print(f"Output directory: {self.output_dir}")
    
    def create_output_directory(self):
        """创建输出目录"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def setup_perspective_transform(self):
        """设置透视变换参数"""
        pc_pts = np.array([[-1, -5.14], [-1, -7.95], [2, -5.14], [2, -7.95]], dtype=np.float32)
        pixel_pts = np.array([[250, 650], [250, 609], [900, 650], [900, 601]], dtype=np.float32)
        self.h_matrix = cv2.getPerspectiveTransform(pc_pts, pixel_pts)
    
    def initialize_video_writer(self, video_key, frame_shape):
        """初始化视频写入器"""
        if video_key in self.video_writers:
            return
        
        video_filename = self.output_videos[video_key]
        video_path = os.path.join(self.output_dir, video_filename)
        
        height, width = frame_shape[:2]
        
        writer = cv2.VideoWriter(
            video_path,
            self.video_fourcc,
            self.video_fps,
            (width, height)
        )
        
        if not writer.isOpened():
            print(f"Failed to open video writer for {video_path}")
            return
        
        self.video_writers[video_key] = writer
        self.frame_counters[video_key] = 0
        print(f"Initialized video writer: {video_path} ({width}x{height})")
    
    def write_frame_to_video(self, frame, video_key):
        """将帧写入视频"""
        if frame is None:
            return
        
        # 确保图像是BGR格式
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # 已经是BGR格式，直接使用
            pass
        elif len(frame.shape) == 2:
            # 灰度图转BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # 初始化视频写入器
        if video_key not in self.video_writers:
            self.initialize_video_writer(video_key, frame.shape)
        
        # 写入帧
        if video_key in self.video_writers:
            self.video_writers[video_key].write(frame)
            self.frame_counters[video_key] += 1
            
            if self.frame_counters[video_key] % 100 == 0:
                print(f"{self.output_videos[video_key]}: {self.frame_counters[video_key]} frames")
    
    def close_all_writers(self):
        """关闭所有视频写入器"""
        for video_key, writer in self.video_writers.items():
            if writer is not None:
                writer.release()
                print(f"Closed video writer: {self.output_videos[video_key]}")
        self.video_writers.clear()
    
    def process_image(self, cv_image, timestamp):
        """处理图像数据"""
        try:
            with self.data_lock:
                self.latest_image = cv_image.copy()
            
            # 保存原始图像
            self.write_frame_to_video(cv_image, 'original')
            
            # 简单的检测处理（绘制一些示例框）
            detection_result = self.simple_detection(cv_image)
            self.write_frame_to_video(detection_result, 'detection')
            
            # 简单的分割处理（基于颜色的道路检测）
            segmentation_result = self.simple_segmentation(cv_image)
            self.write_frame_to_video(segmentation_result, 'segmentation')
            
            # 尝试融合处理
            self.try_fusion_processing()
            
        except Exception as e:
            print(f"Error processing image: {e}")
    
    def simple_detection(self, image):
        """简单的检测处理（示例）"""
        result = image.copy()
        h, w = result.shape[:2]
        
        # 绘制一些示例检测框
        cv2.rectangle(result, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        cv2.putText(result, "Detection Result", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result
    
    def simple_segmentation(self, image):
        """简单的分割处理（基于颜色的道路检测）"""
        result = image.copy()
        h, w = result.shape[:2]
        
        # 转换到HSV空间进行简单的道路检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义灰色道路的HSV范围
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 200])
        
        # 创建道路掩码
        road_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # 形态学操作
        kernel = np.ones((5,5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        # 在原图上叠加道路区域（绿色）
        result[road_mask > 0] = [0, 255, 0]
        result = cv2.addWeighted(image, 0.7, result, 0.3, 0)
        
        # 拟合梯形
        self.img_trapezoid = self.fit_trapezoid_from_mask(road_mask)
        
        # 绘制梯形
        if self.img_trapezoid is not None:
            cv2.polylines(result, [self.img_trapezoid], True, (255, 0, 0), 2)
            cv2.putText(result, "IMG Trapezoid", tuple(self.img_trapezoid[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.putText(result, "Segmentation Result", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result
    
    def fit_trapezoid_from_mask(self, mask):
        """从掩码拟合梯形"""
        h, w = mask.shape
        
        # 找到掩码中的点
        points = np.where(mask > 0)
        if len(points[0]) == 0:
            # 如果没有检测到道路，返回默认梯形
            return self.get_default_trapezoid(w, h)
        
        y_coords = points[0]
        x_coords = points[1]
        
        # 分为上下两部分
        y_mid = h * 0.7
        
        # 底部点
        bottom_mask = y_coords > y_mid
        if np.any(bottom_mask):
            bottom_x = x_coords[bottom_mask]
            bottom_y = y_coords[bottom_mask]
            bottom_left = np.min(bottom_x)
            bottom_right = np.max(bottom_x)
            bottom_y_avg = int(np.mean(bottom_y))
        else:
            bottom_left, bottom_right = w//4, 3*w//4
            bottom_y_avg = int(h * 0.9)
        
        # 顶部点
        top_mask = y_coords < y_mid
        if np.any(top_mask):
            top_x = x_coords[top_mask]
            top_y = y_coords[top_mask]
            top_left = np.min(top_x)
            top_right = np.max(top_x)
            top_y_avg = int(np.mean(top_y))
        else:
            top_left, top_right = w//3, 2*w//3
            top_y_avg = int(h * 0.3)
        
        # 构建梯形 [bottom_left, bottom_right, top_right, top_left]
        trapezoid = np.array([
            [bottom_left, bottom_y_avg],
            [bottom_right, bottom_y_avg],
            [top_right, top_y_avg],
            [top_left, top_y_avg]
        ], dtype=np.int32)
        
        return trapezoid
    
    def get_default_trapezoid(self, w, h):
        """获取默认梯形"""
        return np.array([
            [w//4, int(h*0.9)],      # bottom_left
            [3*w//4, int(h*0.9)],    # bottom_right
            [2*w//3, int(h*0.3)],    # top_right
            [w//3, int(h*0.3)]       # top_left
        ], dtype=np.int32)
    
    def process_lidar(self, pointcloud_msg, timestamp):
        """处理LiDAR数据"""
        try:
            # 提取点云数据
            points = self.extract_points(pointcloud_msg)
            if points is None or len(points) == 0:
                return
            
            with self.data_lock:
                self.latest_lidar_points = points
            
            # 过滤地面点
            filtered_pts = points[points[:, 2] < -0.7]
            if len(filtered_pts) == 0:
                return
            
            # 寻找角点
            corners = self.find_corners(filtered_pts)
            if not corners:
                return
            
            # 转换为梯形
            self.lidar_trapezoid = self.corners_to_trapezoid(corners)
            
            # 生成透视变换可视化
            if self.latest_image is not None:
                perspective_result = self.create_perspective_visualization(self.latest_image, corners)
                if perspective_result is not None:
                    self.write_frame_to_video(perspective_result, 'lidar_perspective')
            
            # 尝试融合处理
            self.try_fusion_processing()
            
        except Exception as e:
            print(f"Error processing lidar: {e}")
    
    def extract_points(self, pc_msg):
        """提取点云数据"""
        try:
            points = []
            for p in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append([p[0], p[1], p[2]])
            return np.array(points) if points else None
        except Exception as e:
            print(f"Error extracting points: {e}")
            return None
    
    def find_corners(self, pts):
        """寻找角点"""
        if len(pts) == 0:
            return None
        
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        
        corners = {}
        corners['left_top'] = pts[np.argmin((x - x_min)**2 + (y - y_max)**2)]
        corners['left_bottom'] = pts[np.argmin((x - x_min)**2 + (y - y_min)**2)]
        corners['right_top'] = pts[np.argmin((x - x_max)**2 + (y - y_max)**2)]
        corners['right_bottom'] = pts[np.argmin((x - x_max)**2 + (y - y_min)**2)]
        
        return corners
    
    def corners_to_trapezoid(self, corners):
        """将角点转换为梯形"""
        try:
            # 转换角点到像素坐标
            corner_names = ['left_top', 'left_bottom', 'right_top', 'right_bottom']
            trapezoid_points = []
            
            for name in corner_names:
                if name in corners:
                    pt = corners[name]
                    ground_pt = np.array([[pt[0], pt[1]]], dtype=np.float32)
                    pixel_pt = cv2.perspectiveTransform(ground_pt.reshape(1, 1, 2), self.h_matrix)
                    x, y = int(pixel_pt[0, 0, 0]), int(pixel_pt[0, 0, 1])
                    trapezoid_points.append([x, y])
            
            if len(trapezoid_points) == 4:
                # 重新排列为 [bottom_left, bottom_right, top_right, top_left]
                reordered = [
                    trapezoid_points[1],  # left_bottom -> bottom_left
                    trapezoid_points[3],  # right_bottom -> bottom_right
                    trapezoid_points[2],  # right_top -> top_right
                    trapezoid_points[0]   # left_top -> top_left
                ]
                return np.array(reordered, dtype=np.int32)
            
            return None
            
        except Exception as e:
            print(f"Error converting corners to trapezoid: {e}")
            return None
    
    def create_perspective_visualization(self, image, corners):
        """创建透视变换可视化"""
        try:
            result = image.copy()
            
            # 转换角点到像素坐标并绘制
            colors = {'left_top': (0, 255, 0), 'left_bottom': (0, 255, 255), 
                     'right_top': (0, 0, 255), 'right_bottom': (255, 0, 0)}
            
            pixel_corners = {}
            for name, pt in corners.items():
                ground_pt = np.array([[pt[0], pt[1]]], dtype=np.float32)
                pixel_pt = cv2.perspectiveTransform(ground_pt.reshape(1, 1, 2), self.h_matrix)
                x, y = int(pixel_pt[0, 0, 0]), int(pixel_pt[0, 0, 1])
                pixel_corners[name] = (x, y)
                
                # 绘制点
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(result, (x, y), 8, colors.get(name, (255, 255, 255)), -1)
                    cv2.circle(result, (x, y), 10, (255, 255, 255), 2)
            
            # 绘制连线
            if len(pixel_corners) == 4:
                pts = np.array([pixel_corners[name] for name in ['left_top', 'right_top', 'right_bottom', 'left_bottom']], dtype=np.int32)
                cv2.polylines(result, [pts], True, (255, 255, 255), 3)
            
            cv2.putText(result, "LiDAR Perspective", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return result
            
        except Exception as e:
            print(f"Error creating perspective visualization: {e}")
            return None
    
    def try_fusion_processing(self):
        """尝试融合处理"""
        try:
            with self.data_lock:
                img_trap = self.img_trapezoid
                lidar_trap = self.lidar_trapezoid
                current_img = self.latest_image
            
            if img_trap is not None and lidar_trap is not None and current_img is not None:
                # 简单的加权平均融合
                fused_trapezoid = self.weighted_average_trapezoid(img_trap, lidar_trap)
                
                if fused_trapezoid is not None:
                    # 生成融合可视化
                    fusion_result = self.create_fusion_visualization(current_img, img_trap, lidar_trap, fused_trapezoid)
                    if fusion_result is not None:
                        self.write_frame_to_video(fusion_result, 'fused')
            
        except Exception as e:
            print(f"Error in fusion processing: {e}")
    
    def weighted_average_trapezoid(self, trap1, trap2):
        """加权平均融合"""
        weight_img = 0.6
        weight_lidar = 0.4
        fused = weight_img * trap1 + weight_lidar * trap2
        return fused.astype(np.int32)
    
    def create_fusion_visualization(self, image, img_trap, lidar_trap, fused_trap):
        """创建融合可视化"""
        try:
            result = image.copy()
            
            # 绘制图像梯形（绿色）
            if img_trap is not None:
                cv2.polylines(result, [img_trap.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                cv2.putText(result, "IMG", tuple(img_trap[0] + [5, -5]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 绘制LiDAR梯形（蓝色）
            if lidar_trap is not None:
                cv2.polylines(result, [lidar_trap.reshape(-1, 1, 2)], True, (255, 0, 0), 2)
                cv2.putText(result, "LIDAR", tuple(lidar_trap[0] + [5, 15]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 绘制融合梯形（紫色）
            if fused_trap is not None:
                cv2.polylines(result, [fused_trap.reshape(-1, 1, 2)], True, (255, 0, 255), 3)
                cv2.putText(result, "FUSED", tuple(fused_trap[0] + [5, -25]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                
                # 绘制简单的路径矩形
                result = self.draw_simple_path(result, fused_trap)
            
            cv2.putText(result, "Fusion Result", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return result
            
        except Exception as e:
            print(f"Error creating fusion visualization: {e}")
            return None
    
    def draw_simple_path(self, image, trapezoid):
        """绘制简单的路径"""
        try:
            h, w = image.shape[:2]
            
            # 获取梯形的底部和顶部中心点
            bottom_center_x = (trapezoid[0][0] + trapezoid[1][0]) // 2
            top_center_x = (trapezoid[2][0] + trapezoid[3][0]) // 2
            bottom_y = trapezoid[0][1]
            top_y = trapezoid[2][1]
            
            # 简单的滤波
            alpha = 0.1
            self.filtered_bottom_x = alpha * bottom_center_x + (1 - alpha) * self.filtered_bottom_x
            self.filtered_top_x = alpha * top_center_x + (1 - alpha) * self.filtered_top_x
            
            # 绘制路径矩形
            bottom_width = abs(trapezoid[1][0] - trapezoid[0][0])
            path_height = bottom_y - top_y
            
            if path_height > 0:
                overlay = image.copy()
                num_rects = min(30, path_height // 10)
                
                for i in range(num_rects):
                    y_pos = bottom_y - i * 10
                    if y_pos <= top_y:
                        break
                    
                    progress = (i * 10) / path_height
                    width_ratio = max(0.3, 1 - progress * 0.7)
                    rect_width = int(bottom_width * width_ratio)
                    
                    center_x_pos = self.filtered_bottom_x + (self.filtered_top_x - self.filtered_bottom_x) * progress
                    center_x_pos = int(center_x_pos)
                    
                    x1 = center_x_pos - rect_width // 2
                    x2 = center_x_pos + rect_width // 2
                    y1 = y_pos - 5
                    y2 = y_pos + 5
                    
                    x1 = max(0, min(w, x1))
                    x2 = max(0, min(w, x2))
                    y1 = max(0, min(h, y1))
                    y2 = max(0, min(h, y2))
                    
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
                
                result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
                return result
            
            return image
            
        except Exception as e:
            print(f"Error drawing path: {e}")
            return image
    
    def get_bag_info(self):
        """获取bag信息"""
        try:
            with rosbag.Bag(self.bag_path, 'r') as bag:
                info = bag.get_type_and_topic_info()
                topics = info.topics
                
                print("=== ROSBag Information ===")
                print(f"Duration: {bag.get_end_time() - bag.get_start_time():.2f} seconds")
                print(f"Total topics: {len(topics)}")
                
                print("\n=== Available Topics ===")
                image_topics = []
                lidar_topics = []
                
                for topic_name, topic_info in topics.items():
                    print(f"  {topic_name}: {topic_info.message_count} messages ({topic_info.msg_type})")
                    
                    # 分类topics
                    if topic_info.msg_type in ['sensor_msgs/Image', 'sensor_msgs/CompressedImage']:
                        image_topics.append(topic_name)
                    elif topic_info.msg_type == 'sensor_msgs/PointCloud2':
                        lidar_topics.append(topic_name)
                
                print(f"\nFound image topics: {image_topics}")
                print(f"Found lidar topics: {lidar_topics}")
                
                return image_topics, lidar_topics
                
        except Exception as e:
            print(f"Error reading bag info: {e}")
            return [], []
    
    def process_bag(self, start_time=None, end_time=None):
        """处理bag文件"""
        try:
            print("Starting bag processing...")
            start_processing = time.time()
            
            # 获取可用topics
            image_topics, lidar_topics = self.get_bag_info()
            
            if not image_topics and not lidar_topics:
                print("No suitable topics found in bag file!")
                return
            
            all_topics = image_topics + lidar_topics
            print(f"Processing topics: {all_topics}")
            
            with rosbag.Bag(self.bag_path, 'r') as bag:
                bag_start_time = bag.get_start_time()
                bag_end_time = bag.get_end_time()
                
                if start_time is not None:
                    start_time = rospy.Time.from_sec(bag_start_time + start_time)
                else:
                    start_time = rospy.Time.from_sec(bag_start_time)
                
                if end_time is not None:
                    end_time = rospy.Time.from_sec(bag_start_time + end_time)
                else:
                    end_time = rospy.Time.from_sec(bag_end_time)
                
                # 统计消息数量
                total_messages = 0
                for topic, msg, timestamp in bag.read_messages(topics=all_topics,
                                                               start_time=start_time, 
                                                               end_time=end_time):
                    total_messages += 1
                
                print(f"Total messages to process: {total_messages}")
                
                # 处理消息
                processed = 0
                for topic, msg, timestamp in bag.read_messages(topics=all_topics,
                                                               start_time=start_time, 
                                                               end_time=end_time):
                    
                    if topic in image_topics:
                        # 处理图像
                        if msg._type == 'sensor_msgs/Image':
                            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                        elif msg._type == 'sensor_msgs/CompressedImage':
                            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
                        else:
                            continue
                        
                        self.process_image(cv_image, timestamp)
                    
                    elif topic in lidar_topics:
                        # 处理LiDAR
                        self.process_lidar(msg, timestamp)
                    
                    processed += 1
                    if processed % 100 == 0:
                        progress = (processed / total_messages) * 100
                        print(f"Progress: {processed}/{total_messages} ({progress:.1f}%)")
                
                # 关闭所有写入器
                self.close_all_writers()
                
                processing_time = time.time() - start_processing
                print("=== Processing Complete ===")
                print(f"Total time: {processing_time:.2f} seconds")
                print(f"Processed messages: {processed}")
                
                # 显示生成的视频
                print("\n=== Generated Videos ===")
                for video_key, video_name in self.output_videos.items():
                    video_path = os.path.join(self.output_dir, video_name)
                    if os.path.exists(video_path):
                        frames = self.frame_counters.get(video_key, 0)
                        duration = frames / self.video_fps
                        size_mb = os.path.getsize(video_path) / (1024*1024)
                        print(f"  ✓ {video_name}: {frames} frames, {duration:.1f}s, {size_mb:.1f}MB")
                    else:
                        print(f"  ✗ {video_name}: not generated")
                
        except Exception as e:
            print(f"Error processing bag: {e}")
            self.close_all_writers()

def main():
    parser = argparse.ArgumentParser(description='Process ROSBag and generate videos')
    parser.add_argument('--bag_path', default='/workspace/xy_erp_2025-06-10-15-57-53.bag',
                       help='Path to the ROSBag file')
    parser.add_argument('--output_dir', '-o', default='/workspace/results',
                       help='Output directory (default: /workspace/results)')
    parser.add_argument('--start_time', '-s', type=float,
                       help='Start time in seconds from bag start')
    parser.add_argument('--end_time', '-e', type=float,
                       help='End time in seconds from bag start')
    parser.add_argument('--fps', type=int, default=15,
                       help='Output video FPS (default: 15)')
    parser.add_argument('--info_only', action='store_true',
                       help='Only show bag information, do not process')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bag_path):
        print(f"Error: ROSBag file not found: {args.bag_path}")
        return 1
    
    try:
        # 初始化ROS（如果需要）
        try:
            rospy.get_node_uri()
        except:
            rospy.init_node('simple_bag_processor', anonymous=True)
        
        # 创建处理器
        processor = SimpleBagProcessor(args.bag_path, args.output_dir)
        processor.video_fps = args.fps
        
        if args.info_only:
            # 只显示信息
            processor.get_bag_info()
        else:
            # 开始处理
            processor.process_bag(
                start_time=args.start_time,
                end_time=args.end_time
            )
            
            print("=== All Processing Complete ===")
            print(f"Output directory: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())