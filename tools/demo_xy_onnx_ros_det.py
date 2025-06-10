#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import rospy
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import torchvision.transforms as transforms
from ultralytics import YOLO
from threading import Lock, Thread
from collections import deque
import time


# Add your project paths here if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import your custom modules (adjust paths as needed)
try:
    from lib.core.general import non_max_suppression, scale_coords
    from lib.utils import plot_one_box, show_seg_result_xy_ros
except ImportError as e:
    rospy.logwarn(f"Custom module import failed: {e}")
    # Provide fallback functions if imports fail
    def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False):
        """Fallback NMS function"""
        return [torch.empty(0, 6)]
    
    def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
        """Fallback coordinate scaling function"""
        return coords
    
    def plot_one_box(x, img, color=None, label=None, line_thickness=3):
        """Fallback box plotting function"""
        pass
    
    def show_seg_result_xy_ros(*args, **kwargs):
        """Fallback segmentation visualization"""
        return args[1] if len(args) > 1 else None

TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

class YOLOPRealtimeInference:
    def __init__(self, weights_path, yolo_weights_path, conf_thres=0.5, iou_thres=0.5, img_size=640):
        """
        Initialize YOLOP real-time inference node
        
        Args:
            weights_path (str): Path to ONNX model weights
            yolo_weights_path (str): Path to YOLO model weights
            conf_thres (float): Confidence threshold for detection
            iou_thres (float): IoU threshold for NMS
            img_size (int): Input image size for model
        """
        # Initialize ROS node
        rospy.init_node('yolop_realtime_inference_node', anonymous=True)
        
        # Initialize parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Image queue and threading - keep 5 latest images
        self.image_queue = deque(maxlen=5)
        self.image_lock = Lock()
        
        # Frame counter
        self.frame_id = 0
        
        # Minimal performance monitoring (removed detailed stats)
        self.last_stats_print = time.time()
        self.stats_print_interval = 10.0  # Reduced frequency
        
        # Load detection model
        self.det_model = YOLO(yolo_weights_path)
        
        # Load ONNX model for segmentation
        self.ort_session = ort.InferenceSession(weights_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [output.name for output in self.ort_session.get_outputs()]

        rospy.loginfo(f"ONNX input: {self.input_name}")
        rospy.loginfo(f"ONNX outputs: {self.output_names}")
        
        # Initialize image preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])
        
        # Detection classes and colors
        self.names = ['car', 'truck', 'bus', 'person', 'bike', 'motor']
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        
        # ROS subscriber for compressed images
        self.image_sub = rospy.Subscriber(
            "/image_jpeg/compressed", 
            CompressedImage, 
            self.image_callback, 
            queue_size=1,
            buff_size=2**24  # 16MB buffer
        )
        
        # ROS publishers for separate results
        self.detection_pub = rospy.Publisher(
            "/yolop/results_det/compressed", 
            CompressedImage, 
            queue_size=1
        )
        
        self.segmentation_pub = rospy.Publisher(
            "/yolop/results_seg/compressed", 
            CompressedImage, 
            queue_size=1
        )
        
        # Threading control
        self.running = True
        
        # Start processing threads
        self.detection_thread = Thread(target=self.detection_processing_loop, daemon=True)
        self.segmentation_thread = Thread(target=self.segmentation_processing_loop, daemon=True)
        
        rospy.loginfo("YOLOP real-time inference node started, waiting for images...")
    
    def image_callback(self, msg):
        """
        ROS callback function for caching incoming images
        
        Args:
            msg (CompressedImage): ROS compressed image message
        """
        try:
            # Convert ROS compressed image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Store image with thread safety - automatically keeps only latest 5
            with self.image_lock:
                timestamp = msg.header.stamp if hasattr(msg, 'header') else rospy.Time.now()
                self.image_queue.append({
                    'image': cv_image,  # Removed .copy() for speed
                    'timestamp': timestamp,
                    'frame_id': self.frame_id
                })
                self.frame_id += 1
                
        except Exception as e:
            pass  # Silent error handling for speed
    
    def get_latest_image(self):
        """
        Get the latest image from the queue
        
        Returns:
            dict or None: Latest image data or None if queue is empty
        """
        with self.image_lock:
            if len(self.image_queue) > 0:
                return self.image_queue[-1]  # Removed .copy() for speed
            return None
    
    def preprocess_image(self, cv_image):
        """
        Preprocess image for ONNX inference
        
        Args:
            cv_image (np.ndarray): Input OpenCV image
            
        Returns:
            tuple: (processed_tensor, original_image, shapes_info)
        """
        # Resize image while maintaining aspect ratio
        img_height, img_width = cv_image.shape[:2]
        
        # Calculate padding to maintain aspect ratio
        scale = min(self.img_size / img_height, self.img_size / img_width)
        new_height, new_width = int(img_height * scale), int(img_width * scale)
        
        # Resize image
        resized_img = cv2.resize(cv_image, (new_width, new_height))
        
        # Create padded image
        padded_img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (self.img_size - new_width) // 2
        pad_y = (self.img_size - new_height) // 2
        
        # Place resized image in center
        padded_img[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized_img
        
        # Convert BGR to RGB for model input
        rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor_img = self.transform(rgb_img).unsqueeze(0)
        
        # Store shape information for later use
        shapes_info = {
            'original_shape': (img_height, img_width),
            'scale': scale,
            'pad': (pad_x, pad_y),
            'new_shape': (new_height, new_width)
        }
        
        return tensor_img, cv_image, shapes_info
    
    def run_segmentation_inference(self, img_tensor):
        """
        Run ONNX inference for segmentation only
        
        Args:
            img_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            tuple: (driving_area_output, lane_line_output)
        """
        # Convert to numpy for ONNX
        img_np = img_tensor.cpu().numpy().astype(np.float32)
        ort_inputs = {self.input_name: img_np}
        
        # Run inference
        onnx_outputs = self.ort_session.run(self.output_names, ort_inputs)
        
        # Convert outputs back to torch tensors (focus on segmentation outputs)
        da_seg_out = torch.from_numpy(onnx_outputs[4]) if len(onnx_outputs) > 4 else None
        ll_seg_out = torch.from_numpy(onnx_outputs[5]) if len(onnx_outputs) > 5 else None
        
        return da_seg_out, ll_seg_out
    
    def detect_objects(self, image):
        """
        Detect and track objects using YOLO
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            list: List of detections with tracking IDs
        """
        results = self.det_model.track(image, persist=True)
        
        detections = []

        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes
                
                for box in boxes:
                    cls = int(box.cls.item())
                    if cls not in TARGET_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf.item())
                    track_id = int(box.id.item()) if box.id is not None else -1
                    
                    class_name = self.det_model.names[cls] if cls < len(self.det_model.names) else "object"
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': class_name,
                        'track_id': track_id
                    })

        return detections
    
    def postprocess_segmentation(self, da_seg_out, ll_seg_out, original_img, shapes_info):
        """
        Postprocess segmentation results
        
        Args:
            da_seg_out (torch.Tensor): Driving area segmentation output
            ll_seg_out (torch.Tensor): Lane line segmentation output
            original_img (np.ndarray): Original input image
            shapes_info (dict): Shape information for resizing
            
        Returns:
            tuple: (driving_area_mask, lane_line_mask)
        """
        if da_seg_out is None or ll_seg_out is None:
            return None, None
        
        h, w = original_img.shape[:2]
        pad_x, pad_y = shapes_info['pad']
        model_h, model_w = self.img_size, self.img_size
        
        # Crop padding from segmentation outputs
        da_predict = da_seg_out[:, :, pad_y:(model_h-pad_y), pad_x:(model_w-pad_x)]
        ll_predict = ll_seg_out[:, :, pad_y:(model_h-pad_y), pad_x:(model_w-pad_x)]
        
        # Resize to original image size
        da_seg_mask = F.interpolate(da_predict, size=(h, w), mode='bilinear')
        ll_seg_mask = F.interpolate(ll_predict, size=(h, w), mode='bilinear')
        
        # Get final masks
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        
        # Process driving area mask
        da_seg_mask = da_seg_mask - ll_seg_mask
        road_mask = np.zeros_like(da_seg_mask)
        road_mask[da_seg_mask == 1] = 1
        
        return road_mask, ll_seg_mask
    
    def visualize_detection_results(self, img, detections):
        """
        Visualize detection results only
        
        Args:
            img (np.ndarray): Input image
            detections (list): Detection results with tracking
            
        Returns:
            np.ndarray: Visualized image
        """
        # Draw detections with tracking
        if detections:
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                track_id = detection['track_id']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw labels with ID on top, class and confidence below
                if track_id != -1:
                    # Draw ID on top line
                    id_label = f'ID: {track_id}'
                    cv2.putText(img, id_label, (x1, y1 - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw class and confidence on bottom line
                    class_label = f'{class_name}: {confidence:.2f}'
                    cv2.putText(img, class_label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # No tracking ID, just show class and confidence
                    label = f'{class_name}: {confidence:.2f}'
                    cv2.putText(img, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return img
    
    def visualize_segmentation_results(self, img, da_mask, ll_mask):
        """
        Visualize segmentation results only
        
        Args:
            img (np.ndarray): Input image
            da_mask (np.ndarray): Driving area mask
            ll_mask (np.ndarray): Lane line mask
            
        Returns:
            np.ndarray: Visualized image
        """
        if da_mask is not None and ll_mask is not None:
            try:
                return show_seg_result_xy_ros(
                    None, img, (da_mask, ll_mask), 
                    None, None, is_demo=True,
                    draw_path=True, 
                    draw_markers=True,
                    draw_trapezoid=False
                )
            except:
                # Fallback: simple color overlay
                return self.simple_segmentation_overlay(img, da_mask, ll_mask)
        
        return img
    
    def simple_segmentation_overlay(self, img, da_mask, ll_mask):
        """
        Simple segmentation overlay as fallback
        
        Args:
            img (np.ndarray): Input image
            da_mask (np.ndarray): Driving area mask
            ll_mask (np.ndarray): Lane line mask
            
        Returns:
            np.ndarray: Image with segmentation overlay
        """
        result = img.copy()
        
        # Driving area in green (semi-transparent)
        if da_mask is not None:
            da_color = np.array([0, 255, 0], dtype=np.uint8)
            da_overlay = np.zeros_like(img)
            da_overlay[da_mask == 1] = da_color
            result = cv2.addWeighted(result, 0.7, da_overlay, 0.3, 0)
        
        # Lane lines in red
        if ll_mask is not None:
            result[ll_mask == 1] = [0, 0, 255]
        
        return result
    
    def detection_processing_loop(self):
        """
        Processing loop for detection
        """
        rate = rospy.Rate(50)  # Increased to 50 Hz for max speed
        
        while self.running and not rospy.is_shutdown():
            try:
                # Get latest image
                image_data = self.get_latest_image()
                if image_data is None:
                    rate.sleep()
                    continue
                
                cv_image = image_data['image']
                timestamp = image_data['timestamp']
                
                # Perform object detection
                detections = self.detect_objects(cv_image)
                
                # Visualize detection results (modify in-place for speed)
                det_result_img = self.visualize_detection_results(cv_image.copy(), detections)
                
                # Publish detection result
                try:
                    det_msg = self.bridge.cv2_to_compressed_imgmsg(det_result_img, dst_format='jpg')
                    det_msg.header.stamp = timestamp
                    self.detection_pub.publish(det_msg)
                except:
                    pass  # Silent error handling for speed
                
            except:
                pass  # Silent error handling for speed
            
            rate.sleep()
    
    def segmentation_processing_loop(self):
        """
        Processing loop for segmentation
        """
        rate = rospy.Rate(20)  # Increased to 20 Hz
        
        while self.running and not rospy.is_shutdown():
            try:
                # Get latest image
                image_data = self.get_latest_image()
                if image_data is None:
                    rate.sleep()
                    continue
                
                cv_image = image_data['image']
                timestamp = image_data['timestamp']
                
                # Preprocess image for ONNX
                img_tensor, original_img, shapes_info = self.preprocess_image(cv_image)
                
                # Run segmentation inference
                da_seg_out, ll_seg_out = self.run_segmentation_inference(img_tensor)
                
                # Postprocess segmentation
                da_mask, ll_mask = self.postprocess_segmentation(da_seg_out, ll_seg_out, original_img, shapes_info)
                
                # Visualize segmentation results
                seg_result_img = self.visualize_segmentation_results(original_img.copy(), da_mask, ll_mask)
                
                # Publish segmentation result
                try:
                    seg_msg = self.bridge.cv2_to_compressed_imgmsg(seg_result_img, dst_format='jpg')
                    seg_msg.header.stamp = timestamp
                    self.segmentation_pub.publish(seg_msg)
                except:
                    pass  # Silent error handling for speed
                
            except:
                pass  # Silent error handling for speed
            
            rate.sleep()
    
    def print_timing_stats(self):
        """
        Minimal timing statistics (simplified)
        """
        current_time = time.time()
        print(f"Node running... {current_time - self.last_stats_print:.1f}s")
    
    def run(self):
        """
        Main run method - starts processing threads and monitors
        """
        # Start processing threads
        self.detection_thread.start()
        self.segmentation_thread.start()
        
        # Minimal monitoring loop
        try:
            while not rospy.is_shutdown():
                # Minimal stats printing
                current_time = time.time()
                if current_time - self.last_stats_print > self.stats_print_interval:
                    self.print_timing_stats()
                    self.last_stats_print = current_time
                
                rospy.sleep(2.0)  # Check less frequently
                
        except KeyboardInterrupt:
            pass
        
        # Cleanup
        self.running = False


def main():
    # Configuration parameters
    weights_path = "/workspace/onnx_converter/yolopx.onnx"
    yolo_weights_path = "/workspace/onnx_converter/yolov10n.pt"
    conf_thres = 0.3
    iou_thres = 0.45
    img_size = 640
    
    # Check if weights files exist
    if not os.path.exists(weights_path):
        rospy.logerr(f"ONNX model file not found: {weights_path}")
        return
        
    if not os.path.exists(yolo_weights_path):
        rospy.logerr(f"YOLO model file not found: {yolo_weights_path}")
        return
    
    # Create and run inference node
    try:
        inference_node = YOLOPRealtimeInference(
            weights_path=weights_path,
            yolo_weights_path=yolo_weights_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            img_size=img_size
        )
        inference_node.run()
    except Exception as e:
        rospy.logerr(f"Node startup failed: {str(e)}")


if __name__ == '__main__':
    main()