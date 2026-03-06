import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

VIDEO_CHUNK_SIZE = 5  # seconds, size of each video file


class Camera:

    def __init__(
            self,
            camera_name: str,
            rtsp_urls: List[str],
            output_dir: Path = Path('')
    ):
        self.camera_name = camera_name
        self.rtsp_urls = rtsp_urls
        self._output_dir = output_dir

        self._output_dirs = {}
        for index, rtsp_url in enumerate(rtsp_urls):
            dir_for_camera_and_stream = self._output_dir / self.camera_name / f'stream_{index}'
            dir_for_camera_and_stream.mkdir(parents=True, exist_ok=True)
            self._output_dirs[rtsp_url] = dir_for_camera_and_stream

        #
        self.process_dict: Dict[str, subprocess.Popen | None] = defaultdict(lambda: None)

    def record(self):
        for index, rtsp_url in enumerate(self.rtsp_urls):
            dir_for_camera_and_stream = self._output_dirs[rtsp_url]
            self.process_dict[rtsp_url] = threading.Thread(
                target=start_record_thread,
                kwargs={
                    'rtsp_url': rtsp_url,
                    'output_dir': dir_for_camera_and_stream,
                    'cam_obj': self,
                }).start()

    def stop(self):
        for process in self.process_dict.values():
            process.kill()
            process.wait()

    def is_recording(self):
        return any(self.process_dict.values())


def start_record_thread(rtsp_url: str, output_dir: Path, cam_obj: Camera):
    print('start_record_thread', rtsp_url, output_dir)
    last_file_time = None
    max_retries = 3
    retries = 0

    while True:
        if cam_obj.process_dict[rtsp_url] is None:
            cam_obj.process_dict[rtsp_url] = start_recording(rtsp_url, output_dir)
            time.sleep(VIDEO_CHUNK_SIZE + 1)

        newest_file_time = get_newest_file_time(output_dir)
        if last_file_time is None:
            last_file_time = newest_file_time

        print('delta = ', newest_file_time - last_file_time, ', retries=', retries, ',', output_dir)
        if newest_file_time > last_file_time:
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


def get_newest_file_time(directory: Path):
    files = list(directory.glob("*.mp4"))
    if not files:
        return 0
    return max(f.stat().st_mtime for f in files)


def start_recording(rtsp_url: str, output_dir: Path):
    print('start recording')
    output_file_pattern = str(output_dir / "recording_%Y-%m-%d_%H_%M_%S.mp4")

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
            output_dir=Path(kwargs['output_dir'])
        )
        camera.record()
        print('is_recording=', camera.is_recording())

    while True:
        time.sleep(5)
