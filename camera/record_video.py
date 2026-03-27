import subprocess
import threading
import time
from pathlib import Path
from typing import Dict

from camera.file_helpers import get_newest_file_timestamp, VIDEO_FILENAME_PATTERN

processes: Dict[str, subprocess.Popen] = {}  # {rtsp_url: thread}

VIDEO_CHUNK_SIZE: int = 5  # seconds

'''
These threads monitor the files produced by the recording process.
If no file is created, then it is assumed the process died.
This thread will restart it.
'''
threads: Dict[str, threading.Thread] = {}


def start_recording(rtsp_url: str, output_dir: Path):
    print('start recording')
    output_file_dir = output_dir / 'video'
    output_file_dir.mkdir(parents=True, exist_ok=True)

    output_file_pattern = str(output_file_dir / VIDEO_FILENAME_PATTERN)

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

    process = subprocess.Popen(cmd)
    processes[rtsp_url] = process

    # start the thread that will monitor the process
    t = threading.Thread(
        target=_monitor_recording,
        args=(rtsp_url, output_dir)
    )
    threads[rtsp_url] = t


def _monitor_recording(rtsp_url: str, output_dir: Path):
    last_file_time = None
    max_retries = 3
    retries = 0

    while processes[rtsp_url] is not None:

        newest_file_time = get_newest_file_timestamp(output_dir)
        if last_file_time is None:
            last_file_time = newest_file_time

        if time.time() - newest_file_time < VIDEO_CHUNK_SIZE:
            continue

        if newest_file_time > last_file_time:
            last_file_time = newest_file_time
            retries = 0

        elif newest_file_time == last_file_time:
            retries += 1
            print(f'stream is stalled, retrying {max_retries - retries} more time(s)')
            if retries > max_retries:
                print('killing unresponsive recording')
                processes[rtsp_url].kill()
                processes[rtsp_url].wait()
                processes.pop(rtsp_url, None)
                threads.pop(rtsp_url, None)
                retries = 0
                start_recording(rtsp_url, output_dir)
                break  # kill this thread

        time.sleep(1)  # plus 1 to give just a little buffer


def stop_recording(rtsp_url: str):
    process = processes.pop(rtsp_url, None)
    if process:
        process.kill()
        process.wait()
    # note the thread will also be killed
