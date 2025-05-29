## 处理pred结果的.json文件,画图
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


class SimpleKalmanFilter:
    """简易Kalman滤波器用于跟踪梯形顶点"""
    def __init__(self, initial_x=0, initial_y=0, process_noise=1.0, measurement_noise=5.0):
        self.state = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.array([[process_noise, 0, 0, 0], [0, process_noise, 0, 0], 
                          [0, 0, process_noise*0.1, 0], [0, 0, 0, process_noise*0.1]], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        self.max_lateral_change = 8.0  # 横向最大变化限制
        
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]
    
    def update(self, measurement):
        measurement = np.array(measurement, dtype=np.float32)
        if hasattr(self, 'last_position'):
            lateral_change = abs(measurement[0] - self.last_position[0])
            if lateral_change > self.max_lateral_change:
                direction = np.sign(measurement[0] - self.last_position[0])
                measurement[0] = self.last_position[0] + direction * self.max_lateral_change
        
        y_residual = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_residual
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P
        self.last_position = self.state[:2].copy()
        return self.state[:2]
    
    def get_position(self):
        return self.state[:2]

class TrapezoidTracker:
    """梯形顶点跟踪器"""
    def __init__(self):
        self.top_center_filter = None
        self.bottom_center_filter = None
        self.initialized = False
        
    def initialize(self, trapezoid):
        left_bottom, right_bottom, right_top, left_top = trapezoid
        top_center = [(left_top[0] + right_top[0]) / 2, (left_top[1] + right_top[1]) / 2]
        bottom_center = [(left_bottom[0] + right_bottom[0]) / 2, (left_bottom[1] + right_bottom[1]) / 2]
        
        self.top_center_filter = SimpleKalmanFilter(top_center[0], top_center[1], 2.0, 8.0)
        self.bottom_center_filter = SimpleKalmanFilter(bottom_center[0], bottom_center[1], 1.0, 5.0)
        self.initialized = True
        
    def update(self, trapezoid):
        if not self.initialized:
            self.initialize(trapezoid)
        
        left_bottom, right_bottom, right_top, left_top = trapezoid
        observed_top_center = [(left_top[0] + right_top[0]) / 2, (left_top[1] + right_top[1]) / 2]
        observed_bottom_center = [(left_bottom[0] + right_bottom[0]) / 2, (left_bottom[1] + right_bottom[1]) / 2]
        
        predicted_top = self.top_center_filter.predict()
        predicted_bottom = self.bottom_center_filter.predict()
        
        filtered_top = self.top_center_filter.update(observed_top_center)
        filtered_bottom = self.bottom_center_filter.update(observed_bottom_center)
        
        return filtered_top, filtered_bottom

global_tracker = TrapezoidTracker()

def plot_img_and_mask(img, mask, index,epoch,save_dir):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir+"/batch_{}_{}_seg.png".format(epoch,index))

def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False,palette=None,is_demo=False,is_gt=False):
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        color_area[result[0] == 1] = [0, 255, 0]
        color_area[result[1] ==1] = [0, 0, 255]
        color_seg = color_area

    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)
    # img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_segresult.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_segresult.png".format(epoch,index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_seg_gt.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_seg_gt.png".format(epoch,index), img)  
    return img


# def show_seg_result_xy(opt, img, result, index, epoch, save_dir=None, is_ll=False,palette=None,is_demo=False,is_gt=False):
def show_seg_result_xy(opt, img, result, index, epoch, save_dir=None, 
                               is_ll=False, palette=None, is_demo=False, is_gt=False,
                               draw_trapezoid=True, draw_path=True, draw_markers=True):
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:

        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        if 'lad' in opt.show_img:
            color_area[result[0] == 1] = [0, 255, 0]
        if 'das' in opt.show_img:
            try:
                trapezoid, path_info = extract_polygon(result[1], target_class=1, refine=True)
                
                filtered_top, filtered_bottom = draw_enhanced_drivable_path_controlled(
                    color_area, trapezoid, global_tracker,
                    draw_trapezoid=draw_trapezoid,
                    draw_path=draw_path, 
                    draw_center_markers=draw_markers
                )


                color_area[result[1] ==1] = [0, 0, 255]

            

            except ValueError as e:
                print(f"未找到有效像素，使用预测值维持跟踪: {e}")
                
                # 维持跟踪：只进行预测，不更新观测
                if global_tracker.initialized:
                    # 使用跟踪器的预测值
                    predicted_top = global_tracker.top_center_filter.predict()
                    predicted_bottom = global_tracker.bottom_center_filter.predict()
                    
                    # 构建默认梯形来维持绘制
                    image_height, image_width = result[1].shape
                    default_trapezoid = np.array([
                        [predicted_bottom[0] - 100, image_height - 1],  # 左下
                        [predicted_bottom[0] + 100, image_height - 1],  # 右下
                        [predicted_top[0] + 50, predicted_top[1]],      # 右上
                        [predicted_top[0] - 50, predicted_top[1]]       # 左上
                    ], dtype=np.float32)
                    
                    # 使用预测值进行绘制（但不更新跟踪器）
                    if draw_path or draw_markers:
                        draw_vehicle_drivable_path_with_prediction(
                            color_area, predicted_top, predicted_bottom,
                            draw_path=draw_path, draw_markers=draw_markers
                        )
                    
                    if draw_trapezoid:
                        points = np.array(default_trapezoid, dtype=np.int32)
                        cv2.polylines(color_area, [points], isClosed=True, color=(0, 255, 255), thickness=2)
                
                # 显示原始分割结果
                color_area[result[1] == 1] = [0, 0, 255]
                
            except Exception as e:
                print(f"路径绘制失败: {e}")
                color_area[result[1] == 1] = [0, 0, 255]

        color_seg = color_area

    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)
    # img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_segresult.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_segresult.png".format(epoch,index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_seg_gt.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_seg_gt.png".format(epoch,index), img)  
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    #     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)



def draw_vehicle_drivable_path(image: np.ndarray, trapezoid: np.ndarray, 
                              segment_length: int = 20, path_width_ratio: float = 0.8,
                              center_color=(255, 255, 255), border_color=(0, 255, 255),
                              line_thickness=2):
    """
    从车辆中心绘制到梯形上边的可行驶路径
    
    Args:
        image: 输入图像
        trapezoid: 梯形顶点 [左下, 右下, 右上, 左上]
        segment_length: 每个路径段的像素长度
        path_width_ratio: 路径宽度比例
        center_color: 中心线颜色 (白色)
        border_color: 边界线颜色 (蓝色)
        line_thickness: 线条粗细
    """
    left_bottom, right_bottom, right_top, left_top = trapezoid
    
    # 车辆中心：图像底部中线
    vehicle_center_x = image.shape[1] // 2
    vehicle_center_y = image.shape[0] - 1
    
    # 梯形上边中心
    trapezoid_top_center_x = (left_top[0] + right_top[0]) // 2
    trapezoid_top_center_y = (left_top[1] + right_top[1]) // 2
    
    # 计算路径总长度
    total_distance = np.sqrt((trapezoid_top_center_x - vehicle_center_x)**2 + 
                           (trapezoid_top_center_y - vehicle_center_y)**2)
    
    # 计算段数
    num_segments = int(total_distance // segment_length)
    if num_segments < 2:
        num_segments = 2
    
    # 生成路径点
    for i in range(num_segments):
        # 计算当前段的位置比例
        ratio = i / (num_segments - 1)
        
        # 线性插值计算当前点
        current_x = int(vehicle_center_x + ratio * (trapezoid_top_center_x - vehicle_center_x))
        current_y = int(vehicle_center_y + ratio * (trapezoid_top_center_y - vehicle_center_y))
        
        # 计算下一段的点
        next_ratio = min(1.0, (i + 1) / (num_segments - 1))
        next_x = int(vehicle_center_x + next_ratio * (trapezoid_top_center_x - vehicle_center_x))
        next_y = int(vehicle_center_y + next_ratio * (trapezoid_top_center_y - vehicle_center_y))
        
        # 根据梯形形状计算当前位置的路径宽度
        path_width = calculate_path_width_at_position(current_y, trapezoid, image.shape) * path_width_ratio
        
        # 绘制路径段
        if i < num_segments - 1:  # 不绘制最后一个点
            # 绘制中心线 (白色)
            cv2.line(image, (current_x, current_y), (next_x, next_y), 
                    center_color, line_thickness)
            
            # 绘制边界线 (蓝色) - 只在某些段绘制以避免过密
            if i % 3 == 0:  # 每3段绘制一次边界
                # 左边界
                left_offset = int(path_width // 4)
                cv2.line(image, 
                        (current_x - left_offset, current_y), 
                        (next_x - left_offset, next_y), 
                        border_color, line_thickness // 2)
                
                # 右边界
                cv2.line(image, 
                        (current_x + left_offset, current_y), 
                        (next_x + left_offset, next_y), 
                        border_color, line_thickness // 2)


def calculate_path_width_at_position(y: int, trapezoid: np.ndarray, image_shape: tuple) -> float:
    """
    计算在指定y坐标处的路径宽度
    
    Args:
        y: y坐标
        trapezoid: 梯形顶点
        image_shape: 图像形状
        
    Returns:
        该位置的路径宽度
    """
    left_bottom, right_bottom, right_top, left_top = trapezoid
    
    # 计算梯形在该y坐标处的宽度
    bottom_y = max(left_bottom[1], right_bottom[1])
    top_y = min(left_top[1], right_top[1])
    
    if bottom_y == top_y:
        return abs(right_bottom[0] - left_bottom[0])
    
    # 线性插值计算该y坐标处的宽度
    ratio = (y - bottom_y) / (top_y - bottom_y)
    ratio = max(0, min(1, ratio))
    
    bottom_width = abs(right_bottom[0] - left_bottom[0])
    top_width = abs(right_top[0] - left_top[0])
    
    current_width = bottom_width + ratio * (top_width - bottom_width)
    return current_width


def draw_vehicle_drivable_path_with_prediction(image, predicted_top, predicted_bottom, 
                                             draw_path=True, draw_markers=True):
    """使用预测值绘制路径"""
    if draw_path:
        # 绘制预测的中心线
        vehicle_x = image.shape[1] // 2
        vehicle_y = image.shape[0] - 1
        cv2.line(image, (vehicle_x, vehicle_y), 
                (int(predicted_top[0]), int(predicted_top[1])), 
                (255, 255, 255), 2)  # 白色中心线，但较细表示是预测
    
    if draw_markers:
        # 绘制预测的标记点
        vehicle_x = image.shape[1] // 2
        vehicle_y = image.shape[0] - 1
        cv2.circle(image, (vehicle_x, vehicle_y), 8, (0, 0, 255), -1)  # 红点
        cv2.circle(image, (int(predicted_top[0]), int(predicted_top[1])), 6, (0, 255, 0), -1)  # 绿点

def draw_enhanced_drivable_path(image: np.ndarray, trapezoid: np.ndarray):
    """
    绘制增强版可行驶路径（类似openpilot风格）
    
    Args:
        image: 输入图像
        trapezoid: 梯形顶点
    """
    # 1. 绘制梯形轮廓 (黄色)
    points = np.array(trapezoid, dtype=np.int32)
    cv2.polylines(image, [points], isClosed=True, 
                 color=(0, 255, 255), thickness=2)
    
    # 2. 绘制车辆到目标的主路径 (白色中心线 + 蓝色边界)
    draw_vehicle_drivable_path(image, trapezoid, 
                              segment_length=15,
                              path_width_ratio=0.6,
                              center_color=(255, 255, 255),  # 白色中心线
                              border_color=(255, 0, 0),      # 蓝色边界
                              line_thickness=3)
    
    # 3. 绘制车辆位置标记
    vehicle_x = image.shape[1] // 2
    vehicle_y = image.shape[0] - 1
    cv2.circle(image, (vehicle_x, vehicle_y), 8, (0, 0, 255), -1)  # 红色圆点表示车辆
    
    # 4. 绘制目标点标记
    left_bottom, right_bottom, right_top, left_top = trapezoid
    target_x = int((left_top[0] + right_top[0]) // 2)
    target_y = int((left_top[1] + right_top[1]) // 2)
    cv2.circle(image, (target_x, target_y), 6, (100, 255, 100), -1)  # 绿色圆点表示目标


def find_largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    找到梯形的四个顶点
    
    Args:
        mask: 二值掩码
        
    Returns:
        梯形的四个顶点坐标 [左下, 右下, 右上, 左上]
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255
    if len(mask.shape) > 2:
        mask = mask.squeeze()

    # 找到所有mask为True的像素坐标
    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) == 0:
        raise ValueError("未找到任何有效像素")

    # 计算图像中线
    image_width = mask.shape[1]
    image_height = mask.shape[0]
    center_x = image_width // 2
    center_y = image_height // 2  # 纵向中线

    # 分为左半部分和右半部分
    left_mask = x_coords < center_x
    right_mask = x_coords >= center_x
    
    left_x = x_coords[left_mask]
    left_y = y_coords[left_mask]
    right_x = x_coords[right_mask]
    right_y = y_coords[right_mask]

    # 找左上顶点：左半部分中y坐标与纵向中线距离最小的点
    if len(left_y) > 0:
        y_distances = np.abs(left_y - center_y)
        left_top_idx = np.argmin(y_distances)
        left_top = [left_x[left_top_idx], left_y[left_top_idx]]
    else:
        left_top = [0, center_y]

    # 找右上顶点：右半部分中y坐标与纵向中线距离最小的点
    if len(right_y) > 0:
        y_distances = np.abs(right_y - center_y)
        right_top_idx = np.argmin(y_distances)
        right_top = [right_x[right_top_idx], right_y[right_top_idx]]
    else:
        right_top = [image_width-1, center_y]

    # 找左下顶点：左半部分y值最大且x值最小的点
    if len(left_y) > 0:
        max_y_left = np.max(left_y)
        max_y_indices = left_y == max_y_left
        left_bottom_x = left_x[max_y_indices]
        left_bottom_idx = np.argmin(left_bottom_x)  # x值最小
        left_bottom = [left_bottom_x[left_bottom_idx], max_y_left]
    else:
        left_bottom = [0, image_height-1]

    # 找右下顶点：右半部分y值最大且x值最大的点
    if len(right_y) > 0:
        max_y_right = np.max(right_y)
        max_y_indices = right_y == max_y_right
        right_bottom_x = right_x[max_y_indices]
        right_bottom_idx = np.argmax(right_bottom_x)  # x值最大
        right_bottom = [right_bottom_x[right_bottom_idx], max_y_right]
    else:
        right_bottom = [image_width-1, image_height-1]

    # 返回四个顶点：[左下, 右下, 右上, 左上]
    trapezoid = np.array([
        left_bottom,   # 左下
        right_bottom,  # 右下  
        right_top,     # 右上
        left_top       # 左上
    ], dtype=np.float32)
    
    return trapezoid

def fit_trapezoid(contour_points: np.ndarray, image_shape) -> np.ndarray:
        """
        将轮廓拟合成下长上短的梯形
        
        Args:
            contour_points: 轮廓点集
            image_shape: 图像形状 (height, width)
            
        Returns:
            梯形的四个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        height, width = image_shape
        
        # 图像中线
        mid_x = width // 2   # 水平中线，用于分左右
        mid_y = height // 2  # 垂直中线，用作y坐标基线
        
        # 将点按y坐标排序
        sorted_points = contour_points[np.argsort(contour_points[:, 1])]
        
        # 分为上半部分和下半部分
        upper_points = sorted_points[sorted_points[:, 1] < mid_y]
        lower_points = sorted_points[sorted_points[:, 1] >= mid_y]
        
        if len(upper_points) == 0 or len(lower_points) == 0:
            # 如果没有上半部分或下半部分，使用简单的边界框方法
            return simple_trapezoid(contour_points, image_shape)
        
        # 计算上边界的左右端点 - 修改部分
        # 分左右区域
        upper_left_region = upper_points[upper_points[:, 0] < mid_x]  # 左半区域
        upper_right_region = upper_points[upper_points[:, 0] >= mid_x] # 右半区域
        
        # 找左上顶点：左半区域中y坐标与中线距离最小的点
        if len(upper_left_region) > 0:
            y_distances = np.abs(upper_left_region[:, 1] - mid_y)
            upper_left = upper_left_region[np.argmin(y_distances)]
        else:
            # 如果左半区域没有点，使用所有上半部分点中x最小的
            upper_left = upper_points[np.argmin(upper_points[:, 0])]
        
        # 找右上顶点：右半区域中y坐标与中线距离最小的点
        if len(upper_right_region) > 0:
            y_distances = np.abs(upper_right_region[:, 1] - mid_y)
            upper_right = upper_right_region[np.argmin(y_distances)]
        else:
            # 如果右半区域没有点，使用所有上半部分点中x最大的
            upper_right = upper_points[np.argmax(upper_points[:, 0])]
        
        # 计算下边界的左右端点（保持原逻辑）
        lower_left = lower_points[np.argmin(lower_points[:, 0])]
        lower_right = lower_points[np.argmax(lower_points[:, 0])]
        
        # 构建梯形（下长上短）
        # 顺序：左下 -> 右下 -> 右上 -> 左上
        trapezoid = np.array([
            lower_left,   # 左下
            lower_right,  # 右下
            upper_right,  # 右上
            upper_left    # 左上
        ], dtype=np.float32)
        
        return trapezoid


def simple_trapezoid(contour_points: np.ndarray, image_shape) -> np.ndarray:
        """
        简单梯形拟合方法（当复杂方法失败时使用）
        
        Args:
            contour_points: 轮廓点集
            image_shape: 图像形状
            
        Returns:
            梯形顶点
        """
        height, width = image_shape
        
        # 计算边界框
        x_min, y_min = np.min(contour_points, axis=0)
        x_max, y_max = np.max(contour_points, axis=0)
        
        # 创建梯形（下宽上窄）
        bottom_width = x_max - x_min
        top_width = bottom_width * 0.6  # 上边是下边的60%
        
        center_x = (x_min + x_max) / 2
        top_x_offset = top_width / 2
        
        trapezoid = np.array([
            [x_min, y_max],                           # 左下
            [x_max, y_max],                           # 右下
            [center_x + top_x_offset, y_min],         # 右上
            [center_x - top_x_offset, y_min]          # 左上
        ], dtype=np.float32)
        
        return trapezoid
    
def refine_trapezoid(trapezoid: np.ndarray, contour_points: np.ndarray) -> np.ndarray:
        """
        优化梯形以更好地拟合轮廓
        
        Args:
            trapezoid: 初始梯形
            contour_points: 原始轮廓点
            
        Returns:
            优化后的梯形
        """
        # 对每个梯形顶点，找到最近的轮廓点
        refined_trapezoid = trapezoid.copy()
        
        for i, vertex in enumerate(trapezoid):
            # 计算到所有轮廓点的距离
            distances = np.linalg.norm(contour_points - vertex, axis=1)
            # 找到最近的点
            nearest_idx = np.argmin(distances)
            nearest_point = contour_points[nearest_idx]
            
            # 如果距离不太远，就用最近的轮廓点替换
            if distances[nearest_idx] < 20:  # 阈值可调
                refined_trapezoid[i] = nearest_point
        
        return refined_trapezoid
    


def extract_polygon(seg_map: np.ndarray, target_class: int = 1, 
                   refine: bool = True, visualize: bool = False) -> tuple:
    """
    从分割图提取梯形多边形并计算可行驶路径
    
    Args:
        seg_map: 分割图
        target_class: 目标类别值
        refine: 是否优化梯形
        visualize: 是否可视化
        
    Returns:
        (trapezoid, drivable_path): 梯形顶点坐标和可行驶路径信息
    """
    # 1. 预处理：提取目标类别并转换格式
    if target_class == 1 and seg_map.max() <= 1:
        mask = seg_map.astype(np.uint8) * 255
    else:
        mask = (seg_map == target_class).astype(np.uint8) * 255
    
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    
    # 2. 找到梯形顶点
    contour_points = find_largest_contour(seg_map)
    
    # 3. 拟合梯形
    trapezoid = fit_trapezoid(contour_points, seg_map.shape)
    
    # 4. 优化梯形（可选）
    if refine:
        trapezoid = refine_trapezoid(trapezoid, contour_points)
    
    # 5. 计算可行驶路径
    drivable_path = calculate_drivable_path(seg_map, trapezoid)
    
    return trapezoid, drivable_path


def calculate_drivable_path(seg_map: np.ndarray, trapezoid: np.ndarray) -> dict:
    """
    计算可行驶路径信息
    
    Args:
        seg_map: 分割图
        trapezoid: 梯形顶点 [左下, 右下, 右上, 左上]
        
    Returns:
        包含路径信息的字典
    """
    left_bottom, right_bottom, right_top, left_top = trapezoid
    
    # 以左上角y坐标为基准，找该行的x范围
    top_y = int(left_top[1])
    top_y = max(0, min(top_y, seg_map.shape[0]-1))
    
    if seg_map.max() <= 1:
        top_mask_row = seg_map[top_y, :] > 0
    else:
        top_mask_row = seg_map[top_y, :] == 1
    
    if np.any(top_mask_row):
        top_x_indices = np.where(top_mask_row)[0]
        top_x_min = np.min(top_x_indices)
        top_x_max = np.max(top_x_indices)
    else:
        top_x_min = int(left_top[0])
        top_x_max = int(right_top[0])
    
    # 以左下角y坐标为基准，找该行的x范围
    bottom_y = int(left_bottom[1])
    bottom_y = max(0, min(bottom_y, seg_map.shape[0]-1))
    
    if seg_map.max() <= 1:
        bottom_mask_row = seg_map[bottom_y, :] > 0
    else:
        bottom_mask_row = seg_map[bottom_y, :] == 1
    
    if np.any(bottom_mask_row):
        bottom_x_indices = np.where(bottom_mask_row)[0]
        bottom_x_min = np.min(bottom_x_indices)
        bottom_x_max = np.max(bottom_x_indices)
    else:
        bottom_x_min = int(left_bottom[0])
        bottom_x_max = int(right_bottom[0])
    
    return {
        'top_y': top_y,
        'top_x_min': top_x_min,
        'top_x_max': top_x_max,
        'bottom_y': bottom_y,
        'bottom_x_min': bottom_x_min,
        'bottom_x_max': bottom_x_max,
        'center_x': seg_map.shape[1] // 2
    }


def draw_drivable_path(image: np.ndarray, trapezoid: np.ndarray, path_info: dict, 
                      num_lines: int = 8, line_color=(0, 255, 255), line_thickness=2):
    """
    绘制openpilot风格的可行驶路径
    
    Args:
        image: 输入图像
        trapezoid: 梯形顶点
        path_info: 路径信息
        num_lines: 路径线条数量
        line_color: 线条颜色
        line_thickness: 线条粗细
    """
    top_y = path_info['top_y']
    bottom_y = path_info['bottom_y']
    top_x_min = path_info['top_x_min']
    top_x_max = path_info['top_x_max']
    bottom_x_min = path_info['bottom_x_min']
    bottom_x_max = path_info['bottom_x_max']
    center_x = path_info['center_x']
    
    # 从图像底部中心开始到梯形顶部，绘制若干条路径线
    start_y = image.shape[0] - 1  # 图像底部
    
    # 计算每条线的y坐标
    y_positions = np.linspace(start_y, top_y, num_lines)
    
    for i, y in enumerate(y_positions):
        y = int(y)
        
        # 线性插值计算该y坐标处的x范围
        if bottom_y != top_y:
            ratio = (y - bottom_y) / (top_y - bottom_y)
            ratio = max(0, min(1, ratio))  # 限制在[0,1]范围内
        else:
            ratio = 0.5
        
        # 插值计算左右边界
        left_x = int(bottom_x_min + ratio * (top_x_min - bottom_x_min))
        right_x = int(bottom_x_max + ratio * (top_x_max - bottom_x_max))
        
        # 计算路径宽度（逐渐变窄）
        path_width = right_x - left_x
        line_width = int(path_width * 0.1)  # 路径线宽度为路径宽度的10%
        
        # 绘制左右边界线
        if i < num_lines - 2:  # 不在最顶部绘制
            # 左边界线
            left_start = max(0, left_x)
            left_end = min(left_start + line_width, image.shape[1]-1)
            cv2.line(image, (left_start, y), (left_end, y), line_color, line_thickness)
            
            # 右边界线
            right_start = max(0, right_x - line_width)
            right_end = min(right_x, image.shape[1]-1)
            cv2.line(image, (right_start, y), (right_end, y), line_color, line_thickness)
        
        # 绘制中心虚线（每隔一条绘制）
        if i % 2 == 0 and i < num_lines - 1:
            center_line_x = (left_x + right_x) // 2
            dash_length = line_width // 2
            center_start = max(0, center_line_x - dash_length)
            center_end = min(center_line_x + dash_length, image.shape[1]-1)
            cv2.line(image, (center_start, y), (center_end, y), 
                    (255, 255, 255), line_thickness)




def draw_enhanced_drivable_path_controlled(image: np.ndarray, trapezoid: np.ndarray, 
                                         tracker: TrapezoidTracker, 
                                         draw_trapezoid=True, draw_path=True, 
                                         draw_center_markers=True):
    """可控制的增强版可行驶路径绘制"""
    filtered_top_center, filtered_bottom_center = tracker.update(trapezoid)
    
    if draw_trapezoid:
        points = np.array(trapezoid, dtype=np.int32)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 255), thickness=2)
    
    if draw_path:
        filtered_trapezoid = reconstruct_trapezoid_from_centers(trapezoid, filtered_top_center, filtered_bottom_center)
        draw_vehicle_drivable_path(image, filtered_trapezoid, 
                                  segment_length=15, path_width_ratio=0.6,
                                  center_color=(255, 255, 255), border_color=(255, 0, 0), line_thickness=3)
    
    if draw_center_markers:
        vehicle_x = image.shape[1] // 2  # 图像底部中心x坐标
        vehicle_y = image.shape[0] - 1   # 图像底部y坐标
        # vehicle_x, vehicle_y = int(filtered_bottom_center[0]), int(filtered_bottom_center[1])
        target_x, target_y = int(filtered_top_center[0]), int(filtered_top_center[1])
        cv2.circle(image, (vehicle_x, vehicle_y), 8, (0, 0, 255), -1)  # 红色圆点
        cv2.circle(image, (target_x, target_y), 6, (200, 255, 100), -1)    # 绿色圆点
    
    return filtered_top_center, filtered_bottom_center

def reconstruct_trapezoid_from_centers(original_trapezoid, top_center, bottom_center):
    """根据滤波后的中心点重新构建梯形"""
    left_bottom, right_bottom, right_top, left_top = original_trapezoid
    
    original_top_width = abs(right_top[0] - left_top[0])
    original_bottom_width = abs(right_bottom[0] - left_bottom[0])
    
    new_left_top = [top_center[0] - original_top_width/2, top_center[1]]
    new_right_top = [top_center[0] + original_top_width/2, top_center[1]]
    new_left_bottom = [bottom_center[0] - original_bottom_width/2, bottom_center[1]]
    new_right_bottom = [bottom_center[0] + original_bottom_width/2, bottom_center[1]]
    
    return np.array([new_left_bottom, new_right_bottom, new_right_top, new_left_top], dtype=np.float32)

if __name__ == "__main__":
    pass