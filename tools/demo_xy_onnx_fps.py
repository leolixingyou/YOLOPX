import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result_xy

from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


import torch.nn.functional as F

def resize_input_to_match_model(img, target_size=(640, 640)):
    """将输入图像调整到模型期望的尺寸"""
    if img.shape[-2:] != target_size:
        img = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
    return img


def detect(cfg,opt):

    logger = None
    device = select_device(logger,opt.device)
        
    os.makedirs(opt.save_dir, exist_ok=True)  # 直接创建，如果已存在则忽略
    
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load ONNX model
    ort_session = ort.InferenceSession(opt.weights)
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]

    print(f"ONNX模型输入: {input_name}")
    print(f"ONNX模型输出: {output_names}")

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size

    # Get names and colors
    names = ['car', 'truck', 'bus', 'person', 'bike', 'motor']  # 根据你的模型调整
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    vid_path, vid_writer = None, None
    dummy_input = np.zeros((1, 3, 384, opt.img_size), dtype=np.float32)
    _ = ort_session.run(output_names, {input_name: dummy_input})  # run once

    # FPS计算变量
    fps_counter = 0
    total_inference_time = 0.0
    total_postprocess_time = 0.0
    total_preprocess_time = 0.0
    start_time = time.time()
    
    video_count = 0
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        # 开始计时
        frame_start_time = time.time()
        
        # Preprocess - 预处理计时
        preprocess_start_time = time.time()
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # img = resize_input_to_match_model(img, (640, 640))
        # Inference with ONNX - 推理计时
        img_np = img.cpu().numpy().astype(np.float32)  # 强制转换为float32
        ort_inputs = {input_name: img_np}
        preprocess_end_time = time.time()
        
        inference_start_time = time.time()
        onnx_outputs = ort_session.run(output_names, ort_inputs)
        inference_end_time = time.time()

        # 调试输出（仅第一帧）
        if fps_counter == 0:
            print("ONNX 输出调试:")
            for i, output in enumerate(onnx_outputs):
                print(f"输出 {i}: shape={output.shape}, dtype={output.dtype}, range=[{output.min():.3f}, {output.max():.3f}]")
        
        # Postprocess - 后处理计时
        postprocess_start_time = time.time()
        
        # 解析ONNX输出并转换为torch tensor
        det_out = torch.from_numpy(onnx_outputs[0]).to(device)
        da_seg_out = torch.from_numpy(onnx_outputs[4]).to(device) 
        ll_seg_out = torch.from_numpy(onnx_outputs[5]).to(device)

        inf_out = det_out  # ONNX输出可能格式不同，需要根据实际情况调整

        # Apply NMS
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)

        det=det_pred[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, size=(h ,w ), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze()
        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict,  size=(h ,w ), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze()

        da_seg_mask = da_seg_mask-ll_seg_mask
        road_1 = torch.zeros_like(da_seg_mask)
        # road
        road_1[da_seg_mask == 1] = 1
        da_seg_mask = road_1
        da_seg_mask = da_seg_mask.cpu().numpy()
        ll_seg_mask = ll_seg_mask.cpu().numpy() 
        
        postprocess_end_time = time.time()

        if dataset.mode == 'images':
                    # convert to BGR
                    img_det = img_det[..., ::-1]
                    
                    # 修改为与video模式一致的调用方式
                    img_det = show_seg_result_xy(opt, img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True,
                                                draw_path=True, 
                                                draw_markers=True,
                                                draw_trapezoid=False)

                    if len(det):
                        det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
                        for *xyxy,conf,cls in reversed(det):
                            label_det_pred = f'{conf:.2f}'
                            plot_one_box(xyxy, img_det , label=label_det_pred, color=(0,255,255), line_thickness=2)       
                    
                    cv2.imwrite(save_path,img_det)

        elif dataset.mode == 'video':
            img_det = img_det[..., ::-1]

            
            det = det if 'od' in opt.show_img else None 
            ll_seg_mask = ll_seg_mask if 'lad' in opt.show_img else None 
            da_seg_mask = da_seg_mask if 'das' in opt.show_img else None 

            img_det = show_seg_result_xy(opt, img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True, 
                                         draw_path=True, 
                                         draw_markers=True,
                                         draw_trapezoid=False)

            if det != None:        
                if len(det):
                    det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
                    for *xyxy,conf,cls in reversed(det):
                        label_det_pred = f'{conf:.2f}'
                        plot_one_box(xyxy, img_det , label=label_det_pred, color=(0,255,255), line_thickness=2)       
            
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                
            vid_writer.write(img_det)
            # video_count +=1
            # if video_count > 200:
            #     break
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

        # FPS统计
        frame_end_time = time.time()
        
        # 累积时间
        total_preprocess_time += (preprocess_end_time - preprocess_start_time)
        total_inference_time += (inference_end_time - inference_start_time)
        total_postprocess_time += (postprocess_end_time - postprocess_start_time)
        fps_counter += 1
        
        # 每30帧打印一次FPS统计
        if fps_counter % 30 == 0:
            elapsed_time = frame_end_time - start_time
            avg_fps = fps_counter / elapsed_time
            avg_preprocess_time = total_preprocess_time / fps_counter * 1000  # ms
            avg_inference_time = total_inference_time / fps_counter * 1000  # ms
            avg_postprocess_time = total_postprocess_time / fps_counter * 1000  # ms
            frame_time = (frame_end_time - frame_start_time) * 1000  # ms
            
            print(f"[ONNX] Frame {fps_counter}: "
                  f"FPS={avg_fps:.2f}, "
                  f"Frame={frame_time:.2f}ms, "
                  f"Preprocess={avg_preprocess_time:.2f}ms, "
                  f"Inference={avg_inference_time:.2f}ms, "
                  f"Postprocess={avg_postprocess_time:.2f}ms")

    # 最终FPS统计
    total_time = time.time() - start_time
    final_fps = fps_counter / total_time
    avg_preprocess_ms = total_preprocess_time / fps_counter * 1000
    avg_inference_ms = total_inference_time / fps_counter * 1000
    avg_postprocess_ms = total_postprocess_time / fps_counter * 1000
    
    print(f"\n=== ONNX Model Performance Summary ===")
    print(f"Total frames: {fps_counter}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {final_fps:.2f}")
    print(f"Average preprocess time: {avg_preprocess_ms:.2f}ms")
    print(f"Average inference time: {avg_inference_ms:.2f}ms")
    print(f"Average postprocess time: {avg_postprocess_ms:.2f}ms")
    print(f"Average total time per frame: {(avg_preprocess_ms + avg_inference_ms + avg_postprocess_ms):.2f}ms")
    print('Results saved to %s' % Path(opt.save_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/workspace/onnx_converter/yolopx_384.onnx', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='/workspace/inference/image', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='/workspace/inference/cs_output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--show-img', default= "lad_das", help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)