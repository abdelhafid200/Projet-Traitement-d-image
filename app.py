from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
from cv2 import FileStorage
import numpy as np
from PIL import Image
from lib.function import \
    detect_non_vegetative_regions,\
    detect_disease_zones, \
    prediction
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

methods = {
    "detect_non_vegetative_regions": detect_non_vegetative_regions,
    "detect_disease_zones": detect_disease_zones,
}

RESOURCE_FOLDER = 'templates/resources'
UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def save_image(img, filename):
    image = Image.fromarray(img)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
    return filepath

def save_video(video: FileStorage):
    filename = secure_filename(video.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(filepath)
    return filepath

@app.route('/', methods=['GET'])
def index_view():
    return render_template('index.html', methods_list=methods.keys())

@app.route('/image', methods=['GET'])
def image_view():
    return render_template('image.html', methods_list=methods.keys())

@app.route('/video', methods=['GET'])
def video_view():
    return render_template('video.html', methods_list=methods.keys())

@app.route('/predict', methods=['GET'])
def predict_view():
    return render_template('predict.html', methods_list=methods.keys())

@app.route('/resources/<path:filename>', methods=['GET'])
def serve_static(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/background', methods=['GET'])
def serve_background():
    return send_from_directory("./", 'bg3.jpg')


@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    print(request.form)
    if 'method' not in request.form:
        return jsonify({'error': 'No method provided'})

    file = request.files['image']
    method = request.form["method"]
    print(method)
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    app.config['UPLOAD_FOLDER'] = f"{UPLOAD_FOLDER}/{datetime.now().timestamp()}"

    result = methods[method](image)

    image_urls = {}
    for key, img in result.items():
        filepath = save_image(img, f"{key}.jpg")
        image_urls[key] = request.host_url + filepath

    return jsonify({'result': image_urls})

@app.route('/prediction', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    result = prediction(image)
    return jsonify({'result': result})


def process_video(video_path, method):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{method}_output.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame using selected method
        processed_frame = methods[method](frame)
        
        # Write processed frame to output video
        out.write(processed_frame['Result'])

    # Release everything if job is finished
    cap.release()
    out.release()

    return output_path

@app.route('/detect-video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file provided'})
    if 'method' not in request.form:
        return jsonify({'error': 'No method provided'})

    file = request.files['video']
    method = request.form["method"]

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    app.config['UPLOAD_FOLDER'] = f"{UPLOAD_FOLDER}/{datetime.now().timestamp()}"
    if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        filepath = save_video(file)

        output_path = process_video(filepath, method)  # Pass the filepath instead of file
        if output_path:
            print({'result': { "origin": filepath, "result": output_path }})
            return jsonify({'result': {
                "origin": filepath,
                "result": output_path
            }})
        else:
            return jsonify({'error': 'Failed to process video'})
    else:
        return jsonify({'error': 'Unsupported file format'})

if __name__ == '__main__':
    app.run(debug=True)
