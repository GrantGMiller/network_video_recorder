# To start the backend server locally run
# cd backend
# pipenv shell
# python main.py
from pathlib import Path
from typing import List

import cv2
from flask import Flask, render_template, Response

import config
from camera import Camera

app = Flask(
    "Camera Server",
    static_folder="./frontend/build",
    static_url_path="/",
)


@app.route('/')
def index():
    return render_template(
        'camera_view.jinja',
        num_cameras=len(cameras)
    )


@app.route("/camera/<cam_index>/get_latest_frame")
def camera(cam_index):
    cam = cameras[int(cam_index)]
    # print('cam.last_frame_with_annotations=', cam.last_frame_with_annotations)
    last_frame = cam.get_last_frame(with_annotations=True)
    if last_frame is None:
        return 'no image', 404

    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', last_frame)

    # Convert to bytes
    frame_bytes = buffer.tobytes()

    return Response(frame_bytes, mimetype='image/jpeg')


cameras: List[Camera] = []

if __name__ == "__main__":
    for name, kwargs in config.CAMERAS.items():
        camera = Camera(
            camera_name=name,
            rtsp_url=kwargs['rtsp_url'],
            object_detection_rtsp_url=kwargs.get('object_detection_rtsp_url', None),
            output_dir=Path(kwargs['output_dir'])
        )
        cameras.append(camera)
        # camera.start_recording()
        camera.start_object_detection()
        print('is_recording=', camera.is_recording)
        print('is_object_detection_running=', camera.is_object_detection_running)

    app.run(port=9000, debug=True)
