import datetime
import json
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import cv2
import requests
from ultralytics import YOLO

from helpers import get_datetime_from_filename, VIDEO_FILENAME_PATTERN, \
    get_objects_filename_from_datetime, get_image_filename_from_datetime

VIDEO_CHUNK_SIZE = 5  # seconds, size of each video file


class Camera:
    IMAGE_TIMEOUT = 5  # if no image is requested for IMAGE_TIMEOUT, stop writing them to disk
    IMAGES_PER_SECOND = 1  # how many images per second to capture
    MAX_IMAGE_AGE = 60  # images older than MAX_IMAGE_AGE seconds will be deleted

    def __init__(
            self,
            camera_name: str,
            rtsp_urls: List[str],
            output_dir: Path = Path(''),
            object_detection_rtsp_url: str | None = None,
            webhook_url: str | None = None
    ):
        self.camera_name = camera_name
        self.rtsp_urls = rtsp_urls
        self._output_dir = output_dir
        self.object_detection_rtsp_url = object_detection_rtsp_url
        self.webhook_url = webhook_url

        self._output_dirs = {}
        for index, rtsp_url in enumerate(rtsp_urls):
            dir_for_camera_and_stream = self._output_dir / self.camera_name / f'stream_{index}'
            dir_for_camera_and_stream.mkdir(parents=True, exist_ok=True)
            self._output_dirs[rtsp_url] = dir_for_camera_and_stream

        directory = self._output_dir / self.camera_name / 'objects_detected'
        directory.mkdir(parents=True, exist_ok=True)
        self._detected_objects_dir = directory

        self._last_objects_detected = []

        #

        self.process_dict: Dict[str, subprocess.Popen | None] = defaultdict(lambda: None)

        self.object_detection_model = YOLO("yolov8n.pt")
        self.is_object_detection_running = False

        self.last_frame = None
        self.last_frame_with_annotations = None

        self._capture_images = False  # capture a few images on init
        self._images_dir = self._output_dir / self.camera_name / 'images'
        self._images_dir.mkdir(parents=True, exist_ok=True)

    def start_recording(self):
        for index, rtsp_url in enumerate(self.rtsp_urls):
            dir_for_camera_and_stream = self._output_dirs[rtsp_url]
            self.process_dict[rtsp_url] = threading.Thread(
                target=start_record_thread,
                kwargs={
                    'rtsp_url': rtsp_url,
                    'output_dir': dir_for_camera_and_stream,
                    'cam_obj': self,
                }).start()

    def stop_recording(self):
        for process in self.process_dict.values():
            process.kill()
            process.wait()

    @property
    def is_recording(self):
        return any(self.process_dict.values())

    def _log_object_detected(self, detected_obs: Dict[str, list]):
        print('log objs detected', detected_obs)

        # detect change and send webhook
        newly_detected_objects = []

        for obj_label in detected_obs.keys():
            if obj_label not in self._last_objects_detected:
                # a new object was detected
                print('new obj detected', obj_label)
                newly_detected_objects.append(obj_label)
                self._last_objects_detected.append(obj_label)

        for obj_label in self._last_objects_detected:
            if obj_label not in detected_obs.keys():
                # an object disappeared
                print(obj_label, 'disappeared')
                self._last_objects_detected.remove(obj_label)

        print('newly_detected_objects', newly_detected_objects, 'self.last=', self._last_objects_detected)

        if newly_detected_objects and self.webhook_url:
            requests.post(self.webhook_url, json={
                'newly_detected_objects': newly_detected_objects
            })

        # write to file
        video_output_dir = list(self._output_dirs.values())[0]
        newest_file_timestamp = get_newest_file_timestamp(video_output_dir)
        if not newest_file_timestamp:
            # if there is no video file, then create one obj file per minute
            newest_file_timestamp = datetime.datetime.now().replace(second=0).timestamp()

        newest_file_dt = datetime.datetime.fromtimestamp(newest_file_timestamp)

        filename = self._detected_objects_dir / get_objects_filename_from_datetime(newest_file_dt)

        if filename.exists():
            with open(filename) as f:
                try:
                    old_data = json.load(f)
                except Exception as e:
                    print('f.read()=', f.read())
                    print('filename=', filename)
                    print('newest_file_dt=', newest_file_dt)
                    print('error reading json', e)
                    old_data = {}
        else:
            old_data = {}

        new_data = old_data.copy()
        for obj_label, obj_boxes in detected_obs.items():
            for obj_box in obj_boxes:
                temp_list = old_data.get(obj_label, [])
                temp_list.append({
                    'datetime_iso': datetime.datetime.now().isoformat(),
                    'timestamp': time.time(),
                    'box': obj_box
                })
                new_data[obj_label] = temp_list

        json.dump(new_data, open(filename, 'w'))

    def start_object_detection(self):
        if self.object_detection_rtsp_url:
            threading.Thread(
                target=self._do_object_detection,
            ).start()

    def _do_object_detection(self):
        self.is_object_detection_running = True

        cap = cv2.VideoCapture(self.object_detection_rtsp_url)
        self.object_detection_cap = cap

        while self.is_object_detection_running:

            ret, frame = cap.read()
            if not ret:
                cap.release()
                print('opencv stream stalled, restarting')
                time.sleep(3)

            # we can also limit the kinds of objects detected with
            # results = model(frame, conf=0.7, classes=[0]) # 0 is a person
            obj_detect_results = self.object_detection_model.track(frame, conf=0.7, persist=True)

            detected_objs = defaultdict(list)
            for r in obj_detect_results:
                for box in r.boxes:
                    cls = int(box.cls[0])

                    label = self.object_detection_model.names[cls]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_objs[label].append((x1, y1, x2, y2))

            self._log_object_detected(detected_objs)

            # save the image with object_box and label
            print('capture_images=', self._capture_images)
            self.last_frame = frame
            self.last_frame_with_annotations = frame.copy()

            for r in obj_detect_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = self.object_detection_model.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # draw rectangle
                    cv2.rectangle(
                        self.last_frame_with_annotations,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # draw label text
                    cv2.putText(
                        self.last_frame_with_annotations,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

    def save_img(self, frame, obj_detect_results):
        print('save_img', len(obj_detect_results))
        annotated = frame.copy()

        for r in obj_detect_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.object_detection_model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # draw rectangle
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # draw label text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        save_to_path = self._images_dir / get_image_filename_from_datetime(
            datetime.datetime.now()
        )
        print('save_to_path=', save_to_path)
        cv2.imwrite(
            save_to_path,
            annotated
        )

    def get_latest_image_dt(self):
        ret_dt = None
        now_dt = datetime.datetime.now()
        old_images = []

        for filepath in self._images_dir.iterdir():
            filepath_dt = get_datetime_from_filename(filepath)

            # delete old images
            if now_dt - filepath_dt > datetime.timedelta(seconds=self.MAX_IMAGE_AGE):
                old_images.append(filepath)

            if ret_dt is None or ret_dt < filepath_dt:
                ret_dt = filepath_dt

        for old_img in old_images:
            print('deleting old image', old_img)
            os.remove(old_img)

        return ret_dt

    def stop_object_detection(self):
        self.is_object_detection_running = False

    def send_object_detected_message(self, camera_name, objects_detected):
        '''
        Send a HTTP POST to the webserver to give a live notification of an object detected.
        :param objects_detected:
        :return:
        '''
        if self.webhook_url:
            requests.post(self.webhook_url, json={
                'objects_detected': objects_detected,
                'camera_name': camera_name,
            })

    def send_full_update_to_webhook(self, filename: Path | None = None):
        '''
        If filename is None, then send info about all the files.
        If filename is not None, then send info about only that file.
        :param filename:
        :return:
        '''
        print('sending full update to webhook, filename=', filename)
        full_update = {
            'camera_name': self.camera_name,
            'videos': [],
        }
        # collect metadata about all the videos
        filenames = [filename] if filename else [filename for filename in
                                                 [output_dir.iterdir() for output_dir in self._output_dirs.values()]]
        for filename in filenames:
            if filename.suffix == '.mp4':
                start_dt = get_datetime_from_filename(filename)
                end_dt = (start_dt + datetime.timedelta(seconds=VIDEO_CHUNK_SIZE))

                full_update['videos'].append({
                    'filename': str(filename),
                    'start_dt_iso': start_dt.isoformat(),
                    'end_dt_iso': end_dt.isoformat(),
                    'objects_detected': self.get_objects_detected(start_dt, end_dt),
                })

        print('Sending full update to webhook', full_update)
        if self.webhook_url:
            requests.post(self.webhook_url, json=full_update)

    def get_objects_detected(self, start_dt, end_dt):
        ret = {}
        for filename in self._detected_objects_dir.iterdir():
            if filename.suffix == '.json':
                this_file_dt = get_datetime_from_filename(filename)
                if not start_dt <= this_file_dt < end_dt:
                    # this file does not contain info within the time boundaries
                    continue

                data = json.load(open(filename))
                for label, l in data.items():
                    for detected_object in l:
                        detected_at_dt = datetime.datetime.fromisoformat(detected_object['datetime_iso'])
                        if start_dt <= detected_at_dt <= end_dt:
                            if not label in ret:
                                ret[label] = []
                            ret[label].append(detected_object)

        return ret

    def get_latest_frame_path(self):
        ret_path = None
        ret_dt = None
        for filepath in self._images_dir.iterdir():
            filepath_dt = get_datetime_from_filename(filepath)
            if ret_dt is None or ret_dt < filepath_dt:
                ret_path = filepath
                ret_dt = filepath_dt
        return ret_path


def start_record_thread(rtsp_url: str, output_dir: Path, cam_obj: Camera):
    print('start_record_thread', rtsp_url, output_dir)
    last_file_time = None
    max_retries = 3
    retries = 0

    while True:
        if cam_obj.process_dict[rtsp_url] is None:
            cam_obj.process_dict[rtsp_url] = start_recording(rtsp_url, output_dir)
            time.sleep(VIDEO_CHUNK_SIZE + 1)

        newest_file_time = get_newest_file_timestamp(output_dir)
        if last_file_time is None:
            last_file_time = newest_file_time

        print('delta = ', newest_file_time - last_file_time, ', retries=', retries, ',', output_dir)
        if newest_file_time > last_file_time:
            cam_obj.send_full_update_to_webhook(
                filename=get_newest_file(output_dir)
            )
            last_file_time = newest_file_time
            retries = 0

        elif newest_file_time == last_file_time:
            print('stream is stalled')
            retries += 1
            print('retries=', retries)
            if retries > max_retries:
                print('killing')
                cam_obj.process_dict[rtsp_url].kill()
                cam_obj.process_dict[rtsp_url].wait()
                cam_obj.process_dict[rtsp_url] = None
                retries = 0

        time.sleep(VIDEO_CHUNK_SIZE + 1)  # plus 1 to give just a little buffer


def get_newest_file(directory):
    files = list(directory.glob("*.mp4"))
    if not files:
        return 0
    newest_file: Path | None = None
    for f in files:
        if newest_file is None or f.stat().st_mtime > newest_file.stat().st_mtime:
            newest_file = f
    return newest_file


def get_newest_file_timestamp(directory: Path):
    files = list(directory.glob("*.mp4"))
    if not files:
        return 0
    return max(f.stat().st_mtime for f in files)


def start_recording(rtsp_url: str, output_dir: Path):
    print('start recording')
    output_file_pattern = str(output_dir / VIDEO_FILENAME_PATTERN)

    cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        "-rw_timeout", "5000000",  # 5 seconds without packets = exit
        '-c', 'copy',
        '-f', 'segment',
        '-segment_time', f'{VIDEO_CHUNK_SIZE}',  # time in seconds to chop the video into
        '-reset_timestamps', '1',
        '-strftime', '1',
        str(output_file_pattern)
    ]

    return subprocess.Popen(cmd)


if __name__ == '__main__':
    import config

    for name, kwargs in config.CAMERAS.items():
        camera = Camera(
            camera_name=name,
            rtsp_urls=kwargs['rtsp_urls'],
            object_detection_rtsp_url=kwargs.get('object_detection_rtsp_url', None),
            output_dir=Path(kwargs['output_dir'])
        )
        camera.start_recording()
        camera.start_object_detection()
        print('is_recording=', camera.is_recording)
        print('is_object_detection_running=', camera.is_object_detection_running)

    while True:
        time.sleep(5)
