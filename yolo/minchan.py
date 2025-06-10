import cv2
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker


def det_track_yolo(model):
    cap = cv2.VideoCapture("/workspace/demo/1.mp4")
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        results = model.track(frame, persist=True)[0]  # 추론 + 추적

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes
            for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                print(f"Frame {frame_id} | ID {int(track_id)} | Box: ({x1}, {y1}, {x2}, {y2})")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f'ID:{int(track_id)}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def det_track_byte(model):
    cap = cv2.VideoCapture("/workspace/demo/1.mp4")
    frame_id = 0

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
            score = track[5]  # 필요시 사용
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            print(f"Frame {frame_id} | ID {track_id} | Box: ({x1}, {y1}, {x2}, {y2}) | Score: {score:.2f}")

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = YOLO("./pt/yolov10n.pt")
    # det_track_yolo(model)
    det_track_byte(model)
