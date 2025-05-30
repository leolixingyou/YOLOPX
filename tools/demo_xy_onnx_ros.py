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
from pathlib import Path

# Add your project paths here if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import your custom modules (adjust paths as needed)
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result_xy_ros

class YOLOPInference:
    def __init__(self, weights_path, conf_thres=0.3, iou_thres=0.45, img_size=640):
        """
        Initialize YOLOP inference node
        
        Args:
            weights_path (str): Path to ONNX model weights
            conf_thres (float): Confidence threshold for detection
            iou_thres (float): IoU threshold for NMS
            img_size (int): Input image size for model
        """
        # Initialize ROS node
        rospy.init_node('yolop_inference_node', anonymous=True)
        
        # Initialize parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Load ONNX model
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
            queue_size=1
        )
        
        # ROS publisher for results
        self.result_pub = rospy.Publisher(
            "/yolop/result_image/compressed", 
            CompressedImage, 
            queue_size=1
        )
        
        rospy.loginfo("YOLOP inference node started, waiting for images...")
    
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
    
    def run_inference(self, img_tensor):
        """
        Run ONNX inference
        
        Args:
            img_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            tuple: (detection_output, driving_area_output, lane_line_output)
        """
        # Convert to numpy for ONNX
        img_np = img_tensor.cpu().numpy().astype(np.float32)
        ort_inputs = {self.input_name: img_np}
        
        # Run inference
        onnx_outputs = self.ort_session.run(self.output_names, ort_inputs)
        
        # Convert outputs back to torch tensors
        det_out = torch.from_numpy(onnx_outputs[0])
        da_seg_out = torch.from_numpy(onnx_outputs[4]) if len(onnx_outputs) > 4 else None
        ll_seg_out = torch.from_numpy(onnx_outputs[5]) if len(onnx_outputs) > 5 else None
        
        return det_out, da_seg_out, ll_seg_out
    
    def postprocess_detections(self, det_out, original_img, shapes_info):
        """
        Postprocess detection results
        
        Args:
            det_out (torch.Tensor): Detection output from model
            original_img (np.ndarray): Original input image
            shapes_info (dict): Shape information for coordinate scaling
            
        Returns:
            np.ndarray: Processed detections
        """
        # Apply NMS
        det_pred = non_max_suppression(
            det_out, 
            conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres, 
            classes=None, 
            agnostic=False
        )
        
        detections = det_pred[0] if len(det_pred) > 0 and det_pred[0] is not None else torch.empty(0, 6)
        
        if len(detections):
            # Scale coordinates back to original image size
            img_shape = (self.img_size, self.img_size)  # Model input shape
            original_shape = original_img.shape
            detections[:, :4] = scale_coords(img_shape, detections[:, :4], original_shape).round()
        
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
    
    def visualize_results(self, img, detections, da_mask=None, ll_mask=None, 
                         show_detection=True, show_segmentation=True):
        """
        Visualize inference results on image
        
        Args:
            img (np.ndarray): Input image
            detections (torch.Tensor): Detection results
            da_mask (np.ndarray): Driving area mask
            ll_mask (np.ndarray): Lane line mask
            show_detection (bool): Whether to show detection boxes
            show_segmentation (bool): Whether to show segmentation masks
            
        Returns:
            np.ndarray: Visualized image
        """
        result_img = img.copy()
        
        # Draw segmentation masks if available
        if show_segmentation and da_mask is not None and ll_mask is not None:
            try:
                result_img = show_seg_result_xy_ros(
                    None, result_img, (da_mask, ll_mask), 
                    None, None, is_demo=True,
                    draw_path=True, 
                    draw_markers=True,
                    draw_trapezoid=False
                )
            except Exception as e:
                rospy.logwarn(f"Segmentation visualization error: {e}")
                # Fallback: simple color overlay
                result_img = self.simple_segmentation_overlay(result_img, da_mask, ll_mask)
        
        # Draw detection boxes (optional - currently commented out)
        # if show_detection and len(detections) > 0:
        #     for *xyxy, conf, cls in reversed(detections):
        #         if conf > self.conf_thres:
        #             label = f'{self.names[int(cls)] if int(cls) < len(self.names) else "object"}: {conf:.2f}'
        #             color = self.colors[int(cls) % len(self.colors)]
        #             plot_one_box(xyxy, result_img, label=label, color=color, line_thickness=2)
        
        return result_img
    
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
    
    def image_callback(self, msg):
        """
        ROS callback function for processing incoming images
        
        Args:
            msg (CompressedImage): ROS compressed image message
        """
        try:
            # Convert ROS compressed image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Preprocess image
            img_tensor, original_img, shapes_info = self.preprocess_image(cv_image)
            
            # Run inference
            det_out, da_seg_out, ll_seg_out = self.run_inference(img_tensor)
            
            # Postprocess results
            detections = self.postprocess_detections(det_out, original_img, shapes_info)
            da_mask, ll_mask = self.postprocess_segmentation(da_seg_out, ll_seg_out, original_img, shapes_info)
            
            # Visualize results
            result_img = self.visualize_results(
                original_img, detections, da_mask, ll_mask,
                show_detection=True, show_segmentation=True
            )
            
            # Publish result as compressed image
            try:
                result_msg = self.bridge.cv2_to_compressed_imgmsg(result_img)
                result_msg.header.stamp = msg.header.stamp if hasattr(msg, 'header') else rospy.Time.now()
                self.result_pub.publish(result_msg)
            except Exception as e:
                rospy.logwarn(f"Failed to publish result image: {e}")
            
            # Print detection info
            if len(detections) > 0:
                rospy.loginfo(f"Detected {len(detections)} objects")
            
        except Exception as e:
            rospy.logerr(f"Image processing error: {str(e)}")
    
    def run(self):
        """
        Start the ROS node
        """
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down inference node...")


def main():
    # Configuration parameters
    weights_path = "/workspace/onnx_converter/yolopx.onnx"
    conf_thres = 0.3
    iou_thres = 0.45
    img_size = 640
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        rospy.logerr(f"Model file not found: {weights_path}")
        return
    
    # Create and run inference node
    try:
        inference_node = YOLOPInference(
            weights_path=weights_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            img_size=img_size
        )
        inference_node.run()
    except Exception as e:
        rospy.logerr(f"Node startup failed: {str(e)}")


if __name__ == '__main__':
    main()