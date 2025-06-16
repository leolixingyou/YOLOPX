#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float64MultiArray
from threading import Lock

class PerspectiveTransformProcessor:
    def __init__(self):
        rospy.init_node('perspective_transform_processor', anonymous=True)
        self.bridge = CvBridge()
        self.image_lock = Lock()
        self.points_lock = Lock()
        self.latest_image = None
        self.corner_points = None
        
        # Calculate homography matrix
        pc_pts = np.array([[-1, -5.14], [-1, -7.95], [2, -5.14], [2, -7.95]], dtype=np.float32)
        pixel_pts = np.array([[250, 650], [250, 609], [900, 650], [900, 601]], dtype=np.float32)
        self.h_matrix = cv2.getPerspectiveTransform(pc_pts, pixel_pts)
        
        # ROS subscribers and publishers
        self.image_sub = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.image_cb, queue_size=1)
        self.pc_sub = rospy.Subscriber('/ransac_lidar3D', PointCloud2, self.pc_cb, queue_size=1)
        self.pub = rospy.Publisher('/perspective_img/compressed', CompressedImage, queue_size=1)
        self.corners_pub = rospy.Publisher('/lidar_da_tixing', Float64MultiArray, queue_size=1)
    
    def image_cb(self, msg):
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            with self.image_lock:
                self.latest_image = cv_img.copy()
            self.process_and_publish()
        except Exception as e:
            rospy.logerr(f"Image error: {e}")
    
    def pc_cb(self, msg):
        try:
            pts = self.extract_points(msg)
            if pts is None or len(pts) == 0:
                return
            
            filtered_pts = pts[pts[:, 2] < -0.7]
            if len(filtered_pts) == 0:
                return
            
            corners = self.find_corners(filtered_pts)
            if corners:
                with self.points_lock:
                    self.corner_points = corners
                self.process_and_publish()
        except Exception as e:
            rospy.logerr(f"PC error: {e}")
    
    def extract_points(self, pc_msg):
        try:
            return np.array([[p[0], p[1], p[2]] for p in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)])
        except:
            return None
    
    def find_corners(self, pts):
        if len(pts) == 0:
            return None
        
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        
        corners = {}
        corners['left_top'] = pts[np.argmin((x - x_min)**2 + (y - y_max)**2)]
        corners['left_bottom'] = pts[np.argmin((x - x_min)**2 + (y - y_min)**2)]
        corners['right_top'] = pts[np.argmin((x - x_max)**2 + (y - y_max)**2)]
        corners['right_bottom'] = pts[np.argmin((x - x_max)**2 + (y - y_min)**2)]
        
        if y_max < y_min:
            corners['left_top'], corners['left_bottom'] = corners['left_bottom'], corners['left_top']
            corners['right_top'], corners['right_bottom'] = corners['right_bottom'], corners['right_top']
        
        return corners
    
    def transform_to_pixels(self, corners):
        if not corners:
            return None
        
        pixel_corners = {}
        for name, pt in corners.items():
            ground_pt = np.array([[pt[0], pt[1]]], dtype=np.float32)
            pixel_pt = cv2.perspectiveTransform(ground_pt.reshape(1, 1, 2), self.h_matrix)
            pixel_corners[name] = (int(pixel_pt[0, 0, 0]), int(pixel_pt[0, 0, 1]))
        
        return pixel_corners
    
    def draw_corners(self, img, pixel_corners):
        if not pixel_corners:
            return img
        
        result = img.copy()
        colors = {'left_top': (0, 255, 0), 'left_bottom': (0, 255, 255), 
                 'right_top': (0, 0, 255), 'right_bottom': (255, 0, 0)}
        
        # Draw points
        for name, (x, y) in pixel_corners.items():
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(result, (x, y), 8, colors.get(name, (255, 255, 255)), -1)
                cv2.circle(result, (x, y), 10, (255, 255, 255), 2)
        
        # Draw rectangle
        if len(pixel_corners) == 4:
            try:
                pts = np.array([pixel_corners[name] for name in ['left_top', 'right_top', 'right_bottom', 'left_bottom']], dtype=np.int32)
                cv2.polylines(result, [pts], True, (255, 255, 255), 3)
            except:
                pass
        
        return result
    
    def publish_corners(self, corners):
        if not corners:
            return
        
        # Create MultiArray message with corner coordinates
        msg = Float64MultiArray()
        data = []
        
        # Add corners in order: left_top, left_bottom, right_top, right_bottom
        for name in ['left_top', 'left_bottom', 'right_top', 'right_bottom']:
            if name in corners:
                pt = corners[name]
                data.extend([pt[0], pt[1], pt[2]])  # x, y, z
        
        msg.data = data
        self.corners_pub.publish(msg)
    def process_and_publish(self):
        try:
            with self.image_lock:
                img = self.latest_image.copy() if self.latest_image is not None else None
            with self.points_lock:
                corners = self.corner_points.copy() if self.corner_points is not None else None
            
            if img is None or corners is None:
                return
            
            # Publish corner points as MultiArray
            self.publish_corners(corners)
            
            pixel_corners = self.transform_to_pixels(corners)
            result_img = self.draw_corners(img, pixel_corners)
            
            msg = self.bridge.cv2_to_compressed_imgmsg(result_img, dst_format='jpg')
            msg.header.frame_id = "camera"
            self.pub.publish(msg)
        except Exception as e:
            rospy.logerr(f"Process error: {e}")
    
    def run(self):
        rospy.spin()

def main():
    try:
        processor = PerspectiveTransformProcessor()
        processor.run()
    except Exception as e:
        rospy.logerr(f"Main error: {e}")

if __name__ == '__main__':
    main()