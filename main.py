# To start the backend server locally run
# cd backend
# pipenv shell
# python main.py
import datetime
from pathlib import Path
from typing import List

import cv2
from flask import Flask, render_template, Response, jsonify, request, send_file

import config
from camera import Camera
from helpers import get_datetime_from_filename, is_recording_filename, is_object_detection_filename

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


@app.route('/get_all_cameras')
def get_all_cameras():
    ret = []
    for cam in cameras:
        ret.append(cam.name)
    return jsonify(ret)


@app.route('/get_videos')
def get_videos():
    start_dt_iso = request.args.get('start_dt_iso', datetime.datetime.now().isoformat())
    end_dt_iso = request.args.get('end_dt_iso', datetime.datetime.now().isoformat())

    start_dt = datetime.datetime.fromisoformat(start_dt_iso)
    end_dt = datetime.datetime.fromisoformat(end_dt_iso)

    camera_name = request.args.get('camera_name')
    if camera_name is None:
        return 'please supply a http param like "?camera_name=Camera 1"', 500

    recordings_directory = Path('videos') / camera_name / 'recordings'
    detected_objs_directory = Path('videos') / camera_name / 'objects_detected'

    ret = {
        'start_dt_iso': start_dt_iso,
        'end_dt_iso': end_dt_iso,
        'recordings': [],
        'objects_detected': []
    }

    for file in recordings_directory.iterdir():
        print('file=', file)
        if file.is_file():
            file_dt = get_datetime_from_filename(file)
            if start_dt <= file_dt <= end_dt:
                ret['recordings'].append(file.name)

    for file in detected_objs_directory.iterdir():
        print('file=', file)
        if file.is_file():
            file_dt = get_datetime_from_filename(file)
            if start_dt <= file_dt <= end_dt:
                ret['objects_detected'].append(file.name)

    return jsonify(ret)


@app.route('/get_file')
def get_file():
    filename = request.args.get('filename')
    camera_name = request.args.get('camera_name')
    if camera_name is None:
        return 'please supply a http param like "?camera_name=Camera 1"', 500

    recordings_directory = Path('videos') / camera_name / 'recordings'
    detected_objs_directory = Path('videos') / camera_name / 'objects_detected'

    file_to_send = None
    if is_recording_filename(filename):
        file_to_send = recordings_directory / filename
        if not file_to_send.exists():
            return 'File not found', 404

    elif is_object_detection_filename(filename):
        file_to_send = detected_objs_directory / filename
        if not file_to_send.exists():
            return 'File not found', 404

    return send_file(file_to_send)


cameras: List[Camera] = []

if __name__ == "__main__":
    for name, kwargs in config.CAMERAS.items():
        camera = Camera(
            name=name,
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
