import datetime
import json
import time
from pathlib import Path

from camera import record_video
from camera.object_detection import get_last_frame, start_object_detection, stop_object_detection
from camera.record_video import stop_recording
from helpers import get_objects_filename_from_datetime


class Camera:
    def __init__(
            self,
            name: str,
            rtsp_url: str,
            output_dir: Path = Path(''),
            object_detection_rtsp_url: str | None = None,
    ):
        self.name = name
        self.rtsp_url = rtsp_url
        self.object_detection_rtsp_url = object_detection_rtsp_url
        self.output_dir = output_dir

        self.is_recording = False
        self.is_object_detection_running = False

    def start_recording(self):
        self.is_recording = True
        record_video.start_recording(
            rtsp_url=self.rtsp_url,
            output_dir=self.output_dir / self.name / 'recordings',
        )

    def stop_recording(self):
        self.is_recording = False
        record_video.stop_recording(self.rtsp_url)

    def start_object_detection(self):
        self.is_object_detection_running = True
        start_object_detection(
            self.object_detection_rtsp_url,
            self.on_new_objects_detected
        )

    def on_new_objects_detected(self, new_objs):
        directory = self.output_dir / self.name / 'objects_detected'
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

        filepath = directory / get_objects_filename_from_datetime(
            datetime.datetime.now().replace(microsecond=0, second=0)
        )
        if filepath.exists():
            try:
                data = json.load(open(filepath, 'r'))
            except json.decoder.JSONDecodeError:
                data = {}
        else:
            data = {}

        dt_now_iso = datetime.datetime.now().isoformat()
        for obj_name, positions in new_objs.items():
            if obj_name not in data:
                data[obj_name] = []
            data[obj_name].append({
                'positions': positions,
                'timestamp': time.time(),
                'datetime_iso': dt_now_iso
            })

        json.dump(data, open(filepath, 'w'))

    def stop_object_detection(self):
        self.is_object_detection_running = False
        stop_object_detection(self.object_detection_rtsp_url)

    def get_last_frame(self, with_annotations: bool = True):
        return get_last_frame(self.object_detection_rtsp_url, with_annotations)

    def __del__(self):
        stop_recording(
            self.rtsp_url)
        stop_object_detection(
            self.object_detection_rtsp_url)
