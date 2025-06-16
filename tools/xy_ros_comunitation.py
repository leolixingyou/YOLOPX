#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import MultiArrayDimension, Int16MultiArray
from threading import Thread, Lock
from collections import deque
import time
import numpy as np

class ROSImageNode:
    def __init__(self, processor):
        rospy.init_node('yolop_inference_node', anonymous=True)
        
        self.processor = processor
        self.bridge = CvBridge()
        self.image_queue = deque(maxlen=3)
        self.image_lock = Lock()
        self.running = True
        
        # Subscribers
        self.image_sub = rospy.Subscriber(
            "/image_jpeg/compressed", 
            CompressedImage, 
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # Publishers
        self.det_pub = rospy.Publisher(
            "/yolop/detection/compressed", 
            CompressedImage, 
            queue_size=1
        )
        
        self.seg_pub = rospy.Publisher(
            "/yolop/segmentation/compressed", 
            CompressedImage, 
            queue_size=1
        )
        
        # 新增：梯形数据发布器
        self.trapezoid_pub = rospy.Publisher(
            "/img_da_tixing", 
            Int16MultiArray, 
            queue_size=1
        )
        
        rospy.loginfo("ROS node initialized")
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'passthrough')
            with self.image_lock:
                self.image_queue.append({
                    'image': cv_image,
                    'timestamp': msg.header.stamp if hasattr(msg, 'header') else rospy.Time.now()
                })
        except Exception as e:
            rospy.logwarn(f"Image callback error: {e}")
    
    def get_latest_image(self):
        with self.image_lock:
            return self.image_queue[-1] if self.image_queue else None
    
    def publish_result(self, image, publisher, timestamp):
        try:
            msg = self.bridge.cv2_to_compressed_imgmsg(image, dst_format='jpg')
            msg.header.stamp = timestamp
            publisher.publish(msg)
        except Exception as e:
            rospy.logwarn(f"Publish error: {e}")
    
    def publish_trapezoid(self, trapezoid_points, timestamp):
        """发布梯形数据为Int16MultiArray"""
        try:
            if trapezoid_points is not None:
                # 创建Int16MultiArray消息
                msg = Int16MultiArray()
                
                # 设置数据维度 (4个点，每个点2个坐标)
                dim1 = MultiArrayDimension()
                dim1.label = "points"
                dim1.size = 4
                dim1.stride = 8  # 总共8个数值
                
                dim2 = MultiArrayDimension()
                dim2.label = "coordinates"
                dim2.size = 2
                dim2.stride = 2
                
                msg.layout.dim = [dim1, dim2]
                msg.layout.data_offset = 0
                
                # 展平梯形数据并转换为int16
                # trapezoid_points格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # 发布格式: [x1, y1, x2, y2, x3, y3, x4, y4]
                trapezoid_flat = trapezoid_points.flatten().astype(np.int16)
                msg.data = trapezoid_flat.tolist()
                
                # 发布消息
                self.trapezoid_pub.publish(msg)
                
                rospy.logdebug(f"Published trapezoid: {trapezoid_flat}")
                
        except Exception as e:
            rospy.logwarn(f"Trapezoid publish error: {e}")
    
    def detection_loop(self):
        rate = rospy.Rate(30)
        while self.running and not rospy.is_shutdown():
            image_data = self.get_latest_image()
            if image_data:
                result = self.processor.process_detection(image_data['image'])
                if result is not None:
                    self.publish_result(result, self.det_pub, image_data['timestamp'])
            rate.sleep()
    
    def segmentation_loop(self):
        rate = rospy.Rate(15)
        while self.running and not rospy.is_shutdown():
            image_data = self.get_latest_image()
            if image_data:
                result = self.processor.process_segmentation(image_data['image'])
                if result is not None:
                    self.publish_result(result, self.seg_pub, image_data['timestamp'])
                    
                    # 获取并发布梯形数据
                    trapezoid = self.processor.get_latest_trapezoid()
                    if trapezoid is not None:
                        self.publish_trapezoid(trapezoid, image_data['timestamp'])
                        
            rate.sleep()
    
    def run(self):
        det_thread = Thread(target=self.detection_loop, daemon=True)
        seg_thread = Thread(target=self.segmentation_loop, daemon=True)
        
        det_thread.start()
        seg_thread.start()
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            self.running = False
            rospy.loginfo("Shutting down")