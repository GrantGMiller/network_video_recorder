import time
from pathlib import Path

import config
from camera import Camera

for name, kwargs in config.CAMERAS.items():
    camera = Camera(
        camera_name=name,
        rtsp_urls=kwargs['rtsp_urls'],
        object_detection_rtsp_url=kwargs.get('object_detection_rtsp_url', None),
        output_dir=Path(kwargs['output_dir'])
    )
    # camera.start_recording()
    camera.start_object_detection()
    print('is_recording=', camera.is_recording)
    print('is_object_detection_running=', camera.is_object_detection_running)

#
