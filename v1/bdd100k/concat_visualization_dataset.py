import os
import cv2
import copy
import json
import numpy as np
from tqdm import tqdm

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
print(ROOT_PATH)


def load_and_visualize(img_path, det_path, lane_path, drivable_path, output_path):
    # 读取图像
    img = cv2.imread(img_path)
    copy_img = copy.copy(img)
    lane = cv2.imread(lane_path)
    drivable = cv2.imread(drivable_path) 
    
    # 读取检测结果
    with open(det_path) as f:
        det = json.load(f)
    
    # 画检测框
    colors = {
            'traffic light': (0, 255, 0),
            'traffic sign': (255, 0, 0), 
            'car': (0, 0, 255),
            'pedestrian': (255, 255, 0),
            'truck': (255, 0, 255),
            'bus': (0, 255, 255),
            'rider': (128, 0, 0),
            'motorcycle': (0, 128, 0),
            'bicycle': (0, 0, 128),
            'train': (128, 128, 0),
            'trailer': (128, 0, 128),
            'other vehicle': (0, 128, 128)
            }
   
    for obj in det['frames'][0]['objects']:
        if 'box2d' in obj:
            box = obj['box2d']
            label = obj['category']
            color = colors.get(obj['category'], (255,255,255))
            cv2.putText(img, label, (int(box['x1']), int(box['y1']-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(img, 
                            (int(box['x1']), int(box['y1'])),
                            (int(box['x2']), int(box['y2'])), 
                            color, 2)
    
    # 合并图像
    h, w = img.shape[:2]
    output = np.zeros((h*2, w*2, 3), dtype=np.uint8)
   
    # 排列图像
    output[:h,:w] = img  # 检测框
    output[:h,w:] = lane # 车道线
    output[h:,:w] = drivable # 可行驶区域
    output[h:,w:] = copy_img # 原图
   
    # 保存结果
    cv2.imwrite(output_path, output)

def main():
    splits = ['train', 'val']
    for split in splits:
        img_dir = f'{ROOT_PATH}/labels/det_20/sample_dataset/images/{split}'
        det_dir = f'{ROOT_PATH}/labels/det_20/sample_dataset/det/{split}'
        lane_dir = f'{ROOT_PATH}/labels/det_20/sample_dataset/lane/{split}'
        drivable_dir = f'{ROOT_PATH}/labels/det_20/sample_dataset/drivable/{split}'
        output_dir = f'{ROOT_PATH}/sample_dataset_vis/'

        os.makedirs(output_dir, exist_ok=True)

        img_files = os.listdir(img_dir)

        for img_file in tqdm(img_files[:100]):
            name = img_file.split('.')[0]
           
            img_path = os.path.join(img_dir, f'{name}.jpg')
            det_path = os.path.join(det_dir, f'{name}.json')
            lane_path = os.path.join(lane_dir, f'{name}.png')
            drivable_path = os.path.join(drivable_dir, f'{name}.png')
            output_path = os.path.join(output_dir, f'{name}_vis.jpg')
           
            load_and_visualize(img_path, det_path, lane_path, drivable_path, output_path)

if __name__ == '__main__':
   main()