import cv2
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker


def det_track_yolo(model, output_path="output_yolo.mp4"):
    cap = cv2.VideoCapture("/workspace/demo/1.mp4")
    frame_id = 0
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 设置视频编码器和输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        results = model.track(frame, persist=True)[0]  # 推理 + 追踪

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes
            for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                print(f"Frame {frame_id} | ID {int(track_id)} | Box: ({x1}, {y1}, {x2}, {y2})")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{int(track_id)}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 写入帧到输出视频
        out.write(frame)
        
        # 可选：显示处理进度
        if frame_id % 30 == 0:
            print(f"Processing frame {frame_id}...")
    
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


def det_track_byte(model, output_path="output_byte.mp4"):
    cap = cv2.VideoCapture("/workspace/demo/1.mp4")
    frame_id = 0
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 设置视频编码器和输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    class TrackerArgs:
        track_high_thresh = 0.25  # threshold for the first association
        track_low_thresh = 0.1  # threshold for the second association
        new_track_thresh = 0.25  # threshold for init new track if the detection does not match any tracks
        track_buffer = 30  # buffer to calculate the time when to remove tracks
        match_thresh = 0.8  # threshold for matching tracks
        fuse_score = True  # Whether to fuse confidence scores with the iou distances before matching

    tracker = BYTETracker(TrackerArgs())

    class DetectionResults:
        def __init__(self, boxes, confs, classes):
            self.xywh = boxes
            self.conf = confs
            self.cls = classes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        results = model(frame)[0]
        boxes = results.boxes.xywh.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        det_results = DetectionResults(boxes, confs, classes)

        tracks = tracker.update(det_results, frame.shape[:2], frame_id)

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[0:4])
            track_id = int(track[4])
            score = track[5]  # 需要时使用
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            print(f"Frame {frame_id} | ID {track_id} | Box: ({x1}, {y1}, {x2}, {y2}) | Score: {score:.2f}")

        # 写入帧到输出视频
        out.write(frame)
        
        # 可选：显示处理进度
        if frame_id % 30 == 0:
            print(f"Processing frame {frame_id}...")
    
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


if __name__ == '__main__':
    model = YOLO("./pt/yolov10n.pt")
    # det_track_yolo(model, "output_yolo_tracking.mp4")
    det_track_byte(model, "output_byte_tracking.mp4")