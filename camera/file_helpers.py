import datetime

from pathlib import Path

TIMESTAMP_FORMAT = '%Y-%m-%d %H-%M-%S'
VIDEO_FILENAME_PATTERN = f"{TIMESTAMP_FORMAT} video.mp4"
OBJECTS_FILENAME_PATTERN = f'{TIMESTAMP_FORMAT} objects.json'
IMAGE_FILENAME_PATTERN = f'{TIMESTAMP_FORMAT} image.jpeg'


def get_datetime_from_filename(filename: Path):
    timestamp_str = ((str(filename.stem)
                      .replace(" video", ""))
                     .replace(' objects', '')
                     .replace(' image', ''))

    return datetime.datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)


def get_video_filename_from_datetime(dt: datetime.datetime):
    return dt.strftime(VIDEO_FILENAME_PATTERN)


def get_objects_filename_from_datetime(dt: datetime.datetime):
    return dt.strftime(OBJECTS_FILENAME_PATTERN)


def get_image_filename_from_datetime(dt: datetime.datetime):
    return dt.strftime(IMAGE_FILENAME_PATTERN)


def get_newest_file_timestamp(directory: Path):
    files = list(directory.iterdir())
    if not files:
        return 0
    return max(f.stat().st_mtime for f in files)

def get_newest_file(directory):
    files = list(directory.iterdir())
    if not files:
        return 0
    newest_file: Path | None = None
    for f in files:
        if newest_file is None or f.stat().st_mtime > newest_file.stat().st_mtime:
            newest_file = f
    return newest_file
