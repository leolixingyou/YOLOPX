import cv2
import numpy as np
import onnxruntime as ort
import cv2
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker


def preprocess(frame, input_size=(640, 640)):
   img = cv2.resize(frame, input_size)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img = img.astype(np.float32) / 255.0
   img = np.transpose(img, (2, 0, 1))  # HWC to CHW
   img = np.expand_dims(img, axis=0)  # Add batch dimension
   return img


def postprocess(outputs, original_shape, input_size=(640, 640), conf_thres=0.25):
   pred = outputs[0]

   # (1, N, 6) 형태일 경우 squeeze 해줌
   if len(pred.shape) == 3:
       pred = pred.squeeze(0)

   detections = []
   scale_h, scale_w = original_shape[0] / input_size[1], original_shape[1] / input_size[0]

   for det in pred:
       if float(det[4]) < conf_thres:  # det[4]가 tensor/ndarray일 수 있으므로 float() 처리
           continue

       x1, y1, x2, y2 = det[:4]
       conf = det[4]
       cls = int(det[5])

       cx = (x1 + x2) / 2 * scale_w
       cy = (y1 + y2) / 2 * scale_h
       w = (x2 - x1) * scale_w
       h = (y2 - y1) * scale_h

       detections.append((cx, cy, w, h, conf, cls))

   if len(detections) == 0:
       return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

   boxes = np.array([d[:4] for d in detections], dtype=np.float32)
   confs = np.array([d[4] for d in detections], dtype=np.float32)
   classes = np.array([d[5] for d in detections], dtype=np.int32)
   return boxes, confs, classes


class TrackerArgs:
   track_high_thresh = 0.25
   track_low_thresh = 0.1
   new_track_thresh = 0.25
   track_buffer = 30
   match_thresh = 0.8
   fuse_score = True


class DetectionResults:
   def __init__(self, boxes, confs, classes):
       self.xywh = boxes
       self.conf = confs
       self.cls = classes


def det_track_byte_onnx(session, input_size=(640, 640)):
   cap = cv2.VideoCapture("/workspace/inference/video.mp4")
   frame_id = 0

   # 获取原视频的属性
   fps = int(cap.get(cv2.CAP_PROP_FPS))
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   # 设置视频编码器和输出文件
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter('/workspace/output_tracking.mp4', fourcc, fps, (width, height))

   tracker = BYTETracker(TrackerArgs())

   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break

       frame_id += 1
       img_input = preprocess(frame, input_size)
       ort_inputs = {session.get_inputs()[0].name: img_input}
       outputs = session.run(None, ort_inputs)

       boxes, confs, classes = postprocess(outputs, frame.shape[:2], input_size)
       det_results = DetectionResults(boxes, confs, classes)

       tracks = tracker.update(det_results, frame.shape[:2], frame_id)

       for track in tracks:
           x1, y1, x2, y2 = map(int, track[0:4])
           track_id = int(track[4])
           score = track[5]
           cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
           cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
           print(f"Frame {frame_id} | ID {track_id} | Box: ({x1}, {y1}, {x2}, {y2}) | Score: {score:.2f}")

       # 将处理后的帧写入输出视频
       out.write(frame)

   cap.release()
   out.release()
   print("视频处理完成，输出文件: /workspace/output_tracking.mp4")


if __name__ == '__main__':
   onnx_path = "/workspace/onnx_converter/track_onnx/yolov10n.onnx"
   session = ort.InferenceSession(onnx_path)
   det_track_byte_onnx(session)