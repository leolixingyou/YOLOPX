#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA, Float64MultiArray
from threading import Lock
import time

class PerspectiveTransformProcessor:
    def __init__(self):
        """
        初始化透视变换图像处理器
        """
        # 初始化ROS节点
        rospy.init_node('perspective_transform_processor', anonymous=True)
        
        # 初始化CV桥接器
        self.bridge = CvBridge()
        
        # 线程锁
        self.image_lock = Lock()
        self.points_lock = Lock()
        
        # 存储最新的图像和点云数据
        self.latest_image = None
        self.latest_image_timestamp = None
        self.corner_points = None
        
        # 透视变换矩阵（根据给定的对应关系计算）
        self.perspective_matrix = self.calculate_perspective_matrix()
        
        # 订阅图像话题
        self.image_sub = rospy.Subscriber(
            '/image_jpeg/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1
        )
        
        # 订阅点云话题
        self.pointcloud_sub = rospy.Subscriber(
            '/ransac_lidar3D',
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1
        )
        
        # 发布透视变换后的图像
        self.perspective_pub = rospy.Publisher(
            '/perspective_img/compressed',
            CompressedImage,
            queue_size=1
        )
        

        rospy.loginfo("透视变换处理器已启动")
        rospy.loginfo("订阅图像话题: /image_jpeg/compressed")
        rospy.loginfo("订阅点云话题: /ransac_lidar3D")
        rospy.loginfo("发布结果到: /perspective_img/compressed")
        rospy.loginfo("使用2D地面到2D像素的单应性变换 (Homography)")
    
    def calculate_perspective_matrix(self):
        """
        根据真实透视关系计算2D到2D的单应性矩阵（Homography）
        确保像素坐标都为正数
        
        Returns:
            np.ndarray: 3x3单应性矩阵
        """
        # PC坐标保持不变
        pc_points = np.array([
            [-1, -5.14],     # left_top: x=-1, y=-5.14
            [-1, -7.95],     # left_bottom: x=-1, y=-7.95
            [2, -5.14],      # right_top: x=2, y=-5.14
            [2, -7.95]       # right_bottom: x=2, y=-7.95
        ], dtype=np.float32)
        
        # 调整像素坐标，确保左侧点不会是负数：
        # 根据你的实际检测范围 x:[-2.5, 2.5]，需要更大的左边距
        pixel_points = np.array([
            [250, 650],   # left_top: 左侧x=300 (增加边距), 近处y=650
            [250, 609],   # left_bottom: 左侧x=300, 远处y=609
            [900, 650],   # right_top: 右侧x=700,. 近处y=650
            [900, 601]    # right_bottom: 右侧x=700, 远处y=601
        ], dtype=np.float32)
        
        # 计算单应性矩阵（2D到2D的透视变换）
        homography_matrix = cv2.getPerspectiveTransform(pc_points, pixel_points)
        
        rospy.loginfo("确保正数像素坐标的2D到2D单应性矩阵计算完成:")
        rospy.loginfo(f"Homography Matrix:\n{homography_matrix}")
        rospy.loginfo("调整后的透视变换对应关系:")
        rospy.loginfo("left_top: PC(-1,-5.14) -> Pixel(300,650)")
        rospy.loginfo("left_bottom: PC(-1,-7.95) -> Pixel(300,609)")
        rospy.loginfo("right_top: PC(2,-5.14) -> Pixel(700,650)")
        rospy.loginfo("right_bottom: PC(2,-7.95) -> Pixel(700,601)")
        rospy.loginfo("像素x轴范围: 300-700 (400像素宽度，避免负数)")
        rospy.loginfo("像素y轴范围: 601-650 (约50像素高度)")
        
        return homography_matrix
    
    def image_callback(self, msg):
        """
        图像回调函数
        
        Args:
            msg (CompressedImage): ROS压缩图像消息
        """
        try:
            # 将ROS压缩图像转换为OpenCV格式
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.image_lock:
                self.latest_image = cv_image.copy()
                self.latest_image_timestamp = msg.header.stamp if hasattr(msg, 'header') else rospy.Time.now()
            
            # 如果有点云数据，则处理图像
            self.process_and_publish_image()
                
        except Exception as e:
            rospy.logerr(f"处理图像时出错: {str(e)}")
    
    def pointcloud_callback(self, msg):
        """
        点云回调函数
        
        Args:
            msg (PointCloud2): 点云消息
        """
        try:
            # 从PointCloud2消息中提取点云数据
            points = self.extract_points_from_pointcloud2(msg)
            
            if points is None or len(points) == 0:
                return
            
            # 过滤z轴小于-0.7的点
            filtered_points = self.filter_points_by_z(points, z_threshold=-0.7)
            
            if len(filtered_points) == 0:
                return
            
            # 找到XY平面上的四个顶点
            corner_points = self.find_xy_plane_corners(filtered_points)
            
            if corner_points is not None:
                with self.points_lock:
                    self.corner_points = corner_points
                
                # 如果有图像数据，则处理图像
                self.process_and_publish_image()
            
        except Exception as e:
            rospy.logerr(f"处理点云数据时出错: {str(e)}")
    
    def extract_points_from_pointcloud2(self, pointcloud_msg):
        """
        从PointCloud2消息中提取xyz坐标
        """
        try:
            points_list = []
            
            for point in pc2.read_points(pointcloud_msg, 
                                       field_names=("x", "y", "z"), 
                                       skip_nans=True):
                points_list.append([point[0], point[1], point[2]])
            
            if len(points_list) == 0:
                return None
                
            return np.array(points_list)
            
        except Exception as e:
            rospy.logerr(f"提取点云数据失败: {str(e)}")
            return None
    
    def filter_points_by_z(self, points, z_threshold=-0.7):
        """
        过滤z轴小于阈值的点
        """
        if points is None or len(points) == 0:
            return np.array([])
        
        mask = points[:, 2] < z_threshold
        filtered_points = points[mask]
        
        return filtered_points
    
    def find_xy_plane_corners(self, points):
        """
        在XY平面上找到四个角点：left_top, left_bottom, right_top, right_bottom
        注意：y坐标越大表示越靠"上"（远离原点）
        """
        if points is None or len(points) == 0:
            return None
        
        # 提取x和y坐标
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]
        
        # 找到x和y的最大最小值
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # 对于每个角点，找到最接近的实际点
        corners = {}
        
        # left_top (x_min, y_max) - 左上角：x最小，y最大（最远）
        idx_left_top = np.argmin((x_coords - x_min)**2 + (y_coords - y_max)**2)
        corners['left_top'] = [x_coords[idx_left_top], y_coords[idx_left_top], z_coords[idx_left_top]]
        
        # left_bottom (x_min, y_min) - 左下角：x最小，y最小（最近）
        idx_left_bottom = np.argmin((x_coords - x_min)**2 + (y_coords - y_min)**2)
        corners['left_bottom'] = [x_coords[idx_left_bottom], y_coords[idx_left_bottom], z_coords[idx_left_bottom]]
        
        # right_top (x_max, y_max) - 右上角：x最大，y最大（最远）
        idx_right_top = np.argmin((x_coords - x_max)**2 + (y_coords - y_max)**2)
        corners['right_top'] = [x_coords[idx_right_top], y_coords[idx_right_top], z_coords[idx_right_top]]
        
        # right_bottom (x_max, y_min) - 右下角：x最大，y最小（最近）
        idx_right_bottom = np.argmin((x_coords - x_max)**2 + (y_coords - y_min)**2)
        corners['right_bottom'] = [x_coords[idx_right_bottom], y_coords[idx_right_bottom], z_coords[idx_right_bottom]]
        
        # 根据实际y坐标判断，如果y_max比y_min更负（更小），则需要对调
        if y_max < y_min:
            rospy.logwarn("检测到y坐标系可能颠倒，交换top和bottom")
            # 交换top和bottom
            corners['left_top'], corners['left_bottom'] = corners['left_bottom'], corners['left_top'] 
            corners['right_top'], corners['right_bottom'] = corners['right_bottom'], corners['right_top']
        
        # 验证角点逻辑
        self.validate_corners(corners)
        
        return corners
    
    def validate_corners(self, corners):
        """
        验证角点是否符合矩形逻辑
        """
        if corners is None or len(corners) != 4:
            return
        
        lt = corners['left_top']
        lb = corners['left_bottom'] 
        rt = corners['right_top']
        rb = corners['right_bottom']
        
        rospy.loginfo("=== 角点验证 ===")
        rospy.loginfo(f"left_top: ({lt[0]:.2f}, {lt[1]:.2f})")
        rospy.loginfo(f"left_bottom: ({lb[0]:.2f}, {lb[1]:.2f})")
        rospy.loginfo(f"right_top: ({rt[0]:.2f}, {rt[1]:.2f})")
        rospy.loginfo(f"right_bottom: ({rb[0]:.2f}, {rb[1]:.2f})")
        
        # 验证左侧点x坐标相近
        left_x_diff = abs(lt[0] - lb[0])
        rospy.loginfo(f"左侧x坐标差: {left_x_diff:.3f}")
        
        # 验证右侧点x坐标相近  
        right_x_diff = abs(rt[0] - rb[0])
        rospy.loginfo(f"右侧x坐标差: {right_x_diff:.3f}")
        
        # 验证上侧点y坐标相近
        top_y_diff = abs(lt[1] - rt[1]) 
        rospy.loginfo(f"上侧y坐标差: {top_y_diff:.3f}")
        
        # 验证下侧点y坐标相近
        bottom_y_diff = abs(lb[1] - rb[1])
        rospy.loginfo(f"下侧y坐标差: {bottom_y_diff:.3f}")
        rospy.loginfo("==================")
    
    def transform_points_to_pixels(self, corner_points):
        """
        将地面2D点云坐标转换为像素坐标（使用Homography）
        
        Args:
            corner_points (dict): 四个角点的字典 [x,y,z]
            
        Returns:
            dict: 转换后的像素坐标
        """
        if corner_points is None:
            return None
        
        pixel_corners = {}
        
        for corner_name, point_3d in corner_points.items():
            # 只使用x,y坐标进行2D变换，忽略z
            original_x = point_3d[0]
            original_y = point_3d[1]
            
            # 直接使用原始坐标，不进行x取负的转换
            # 因为单应性矩阵已经处理了坐标系转换
            ground_point_2d = np.array([[original_x, original_y]], dtype=np.float32)
            
            # 应用2D单应性变换
            pixel_point = cv2.perspectiveTransform(ground_point_2d.reshape(1, 1, 2), self.perspective_matrix)
            
            # 提取像素坐标
            pixel_x = int(pixel_point[0, 0, 0])
            pixel_y = int(pixel_point[0, 0, 1])
            
            pixel_corners[corner_name] = (pixel_x, pixel_y)
            
            rospy.loginfo(f"{corner_name}: 地面({original_x:.2f}, {original_y:.2f}) -> 像素({pixel_x}, {pixel_y})")
        
        return pixel_corners
    
    def draw_corners_on_image(self, image, pixel_corners):
        """
        在图像上绘制角点
        
        Args:
            image (np.ndarray): 输入图像
            pixel_corners (dict): 像素角点坐标
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        if pixel_corners is None:
            return image
        
        result_image = image.copy()
        
        # 定义颜色（BGR格式）
        colors = {
            'left_top': (0, 255, 0),      # 绿色 - 左上角
            'left_bottom': (0, 255, 255), # 黄色 - 左下角
            'right_top': (0, 0, 255),     # 红色 - 右上角
            'right_bottom': (255, 0, 0)   # 蓝色 - 右下角
        }
        
        # 绘制角点
        for corner_name, (pixel_x, pixel_y) in pixel_corners.items():
            # 检查坐标是否在图像范围内
            if 0 <= pixel_x < image.shape[1] and 0 <= pixel_y < image.shape[0]:
                color = colors.get(corner_name, (255, 255, 255))
                
                # 绘制圆点
                cv2.circle(result_image, (pixel_x, pixel_y), 8, color, -1)
                
                # 绘制边框
                cv2.circle(result_image, (pixel_x, pixel_y), 10, (255, 255, 255), 2)
                
                # 添加标签
                label = f"{corner_name}"
                cv2.putText(result_image, label, (pixel_x + 15, pixel_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                rospy.logdebug(f"绘制角点 {corner_name} 在像素位置 ({pixel_x}, {pixel_y})")
        
        # 绘制连接线形成矩形
        if len(pixel_corners) == 4:
            try:
                points = []
                # 按顺序连接形成矩形：left_top -> right_top -> right_bottom -> left_bottom -> left_top
                order = ['left_top', 'right_top', 'right_bottom', 'left_bottom']
                
                for corner_name in order:
                    if corner_name in pixel_corners:
                        points.append(pixel_corners[corner_name])
                
                if len(points) == 4:
                    points = np.array(points, dtype=np.int32)
                    # 绘制矩形连接线
                    cv2.polylines(result_image, [points], True, (255, 255, 255), 3)
                    rospy.logdebug("成功绘制四个角点的连接线")
            except Exception as e:
                rospy.logwarn(f"绘制连接线失败: {str(e)}")
        
        return result_image
    
    def process_and_publish_image(self):
        """
        处理图像并发布结果
        """
        try:
            # 获取最新的图像和点云数据
            current_image = None
            current_timestamp = None
            current_corners = None
            
            with self.image_lock:
                if self.latest_image is not None:
                    current_image = self.latest_image.copy()
                    current_timestamp = self.latest_image_timestamp
            
            with self.points_lock:
                if self.corner_points is not None:
                    current_corners = self.corner_points.copy()
            
            # 检查是否都有数据
            if current_image is None or current_corners is None:
                return
            
            # 将3D点转换为像素坐标
            pixel_corners = self.transform_points_to_pixels(current_corners)
            # 在图像上绘制角点
            result_image = self.draw_corners_on_image(current_image, pixel_corners)
            
            # 发布结果图像
            try:
                result_msg = self.bridge.cv2_to_compressed_imgmsg(result_image, dst_format='jpg')
                if current_timestamp:
                    result_msg.header.stamp = current_timestamp
                result_msg.header.frame_id = "camera"
                
                self.perspective_pub.publish(result_msg)
                
                rospy.logdebug("透视变换图像已发布")
                
            except Exception as e:
                rospy.logwarn(f"发布图像失败: {str(e)}")
                
        except Exception as e:
            rospy.logerr(f"处理和发布图像时出错: {str(e)}")
    
    def run(self):
        """
        运行处理器
        """
        rospy.loginfo("透视变换处理器开始运行...")
        
        # 保持节点运行
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("收到中断信号，正在关闭节点...")


def main():
    """
    主函数
    """
    try:
        # 创建处理器实例
        processor = PerspectiveTransformProcessor()
        
        # 运行处理器
        processor.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    except Exception as e:
        rospy.logerr(f"节点启动失败: {str(e)}")


if __name__ == '__main__':
    main()