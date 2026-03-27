from pathlib import Path

from camera import record_video
from camera.object_detection import get_last_frame, start_object_detection, stop_object_detection
from camera.record_video import stop_recording


class Camera:
    def __init__(
            self,
            camera_name: str,
            rtsp_url: str,
            output_dir: Path = Path(''),
            object_detection_rtsp_url: str | None = None,
    ):
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.object_detection_rtsp_url = object_detection_rtsp_url
        self.output_dir = output_dir

        self.is_recording = False
        self.is_object_detection_running = False

    def start_recording(self):
        self.is_recording = True
        record_video.start_recording(
            rtsp_url=self.rtsp_url,
            output_dir=self.output_dir,
        )

    def stop_recording(self):
        self.is_recording = False
        record_video.stop_recording(self.rtsp_url)

    def start_object_detection(self):
        self.is_object_detection_running = True
        start_object_detection(self.object_detection_rtsp_url)

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
