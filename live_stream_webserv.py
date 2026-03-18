# To start the backend server locally run
# cd backend
# pipenv shell
# python main.py
import time
from pathlib import Path
from typing import List

import cv2
from flask import Flask, Response, render_template

import config
from camera import Camera

app = Flask(
    "Camera Server",
    static_folder="./frontend/build",
    static_url_path="/",
)


@app.route('/')
def index():
    return render_template('camera_view.jinja')


@app.route("/camera/<index>/get_latest_frame")
def camera(index):
    camera = cameras[int(index)]

    last_frame = None

    def save_frame(frame):
        print('save_frame(frame=', frame)
        if not frame:
            return
        nonlocal last_frame
        last_frame = frame

    camera.get_latest_frame_jpeg(
        save_frame
    )

    while not last_frame:
        print('waiting for frame...')
        time.sleep(0.5)

    _, jpeg = cv2.imencode(".jpg", last_frame)
    return Response(
        (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
        , mimetype="multipart/x-mixed-replace; boundary=frame"
    )


cameras: List[Camera] = []

if __name__ == "__main__":
    for name, kwargs in config.CAMERAS.items():
        camera = Camera(
            camera_name=name,
            rtsp_urls=kwargs['rtsp_urls'],
            object_detection_rtsp_url=kwargs.get('object_detection_rtsp_url', None),
            output_dir=Path(kwargs['output_dir'])
        )
        cameras.append(camera)
        camera.start_recording()
        camera.start_object_detection()
        print('is_recording=', camera.is_recording)
        print('is_object_detection_running=', camera.is_object_detection_running)

    app.run(port=9000, debug=True)
