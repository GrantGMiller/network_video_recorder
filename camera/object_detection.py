import threading
import time
from collections import defaultdict
from typing import Dict, Any, Callable, Optional

import cv2
from ultralytics import YOLO

obj_detection_model = YOLO("yolov8n.pt")

motion_detection_model: Dict[str, Any] = {}

last_frame: Dict[str, Any] = {}  # {rtsp_url: Frame}
last_frame_with_annotation: Dict[str, Any] = {}
is_recording: Dict[str, bool] = defaultdict(lambda: False)


def start_object_detection(rtsp_url, on_objects_detected: Optional[Callable[[list, ], None]] = None):
    is_recording[rtsp_url] = True
    t = threading.Thread(
        target=_start_detection_loop,
        args=(rtsp_url, on_objects_detected),
    )
    t.start()


def stop_object_detection(rtsp_url):
    is_recording.pop(rtsp_url)
    motion_detection_model.pop(rtsp_url)


def _start_detection_loop(rtsp_url, on_objects_detected):
    cap = cv2.VideoCapture(rtsp_url)
    motion_detection_model[rtsp_url] = cv2.createBackgroundSubtractorMOG2(
        history=500,  # Number of frames for background model
        varThreshold=50,  # Sensitivity (lower = more sensitive)
        detectShadows=True  # Detect and mark shadows
    )

    last_objects_detected = []
    while is_recording.get(rtsp_url, None):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            print('opencv stream stalled, restarting')
            time.sleep(3)

        last_frame[rtsp_url] = frame
        last_frame_with_annotation[rtsp_url] = frame.copy()

        detected_objs = draw_label_and_boxes(rtsp_url, frame)
        detected_objs.update(draw_motion_boxes(rtsp_url, frame))

        # call on_objects_detected here
        objs_detected_this_frame = set(detected_objs.keys())

        new_objs = []
        for obj in objs_detected_this_frame:
            if obj not in last_objects_detected:
                new_objs.append(obj)
                last_objects_detected.append(obj)

        for obj in last_objects_detected:
            if obj not in objs_detected_this_frame:
                # this object was detected, but is no longer
                last_objects_detected.remove(obj)

        if new_objs:
            print('newly detected objects: {}'.format(new_objs))
            if on_objects_detected:
                on_objects_detected(new_objs)


def draw_label_and_boxes(rtsp_url, frame):
    detected_objs = defaultdict(list)

    obj_detect_results = obj_detection_model.track(frame, conf=0.7, persist=True)

    for r in obj_detect_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = obj_detection_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # draw rectangle
            cv2.rectangle(
                last_frame_with_annotation[rtsp_url],
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # draw label text
            cv2.putText(
                last_frame_with_annotation[rtsp_url],
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            detected_objs[label].append((x1, y1, x2, y2))

    return detected_objs


def draw_motion_boxes(rtsp_url, frame):
    detected_objs = defaultdict(list)

    # Apply background subtraction to get foreground mask
    fg_mask = motion_detection_model[rtsp_url].apply(frame)
    # Remove noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes

    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Skip small contours (noise)
            continue

        # Draw bounding box around moving object
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            last_frame_with_annotation[rtsp_url],
            f"Motion ({area:.0f}px)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 2
        )
        detected_objs['motion'].append((x, y, w, h))

    return detected_objs


def get_last_frame(rtsp_url: str, with_annotations: bool = True):
    if with_annotations:
        return last_frame_with_annotation.get(rtsp_url, None)
    else:
        return last_frame.get(rtsp_url, None)
