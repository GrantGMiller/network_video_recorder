import datetime
from pathlib import Path

TIMESTAMP_FORMAT = '%Y-%m-%d %H-%M-%S'
VIDEO_FILENAME_PATTERN = f"{TIMESTAMP_FORMAT} video.mp4"
OBJECTS_FILENAME_PATTERN = f'{TIMESTAMP_FORMAT} objects.json'


def get_datetime_from_filename(filename: Path):
    timestamp_str = (str(filename.stem).replace(" video", "")).replace(' objects', '')

    return datetime.datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)


def get_video_filename_from_datetime(dt: datetime.datetime):
    return dt.strftime(VIDEO_FILENAME_PATTERN)


def get_objects_filename_from_datetime(dt: datetime.datetime):
    return dt.strftime(OBJECTS_FILENAME_PATTERN)
