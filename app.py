import pandas as pd
import torch
from flask import Flask, Response, jsonify
import cv2
import mediapipe as mp
import torchvision.transforms as transforms
from pytorchvideo.models.hub import slowfast_r50
from routes import register_routes
from threading import Thread, Lock
import logging
import notifications
import numpy as np
import time
import os
from PIL import Image

app = Flask(__name__)
app.secret_key = 'supersecretkey'
register_routes(app)

cameras = []
lock = Lock()
dangerous_actions_detected = []

# Путь к файлу с метками классов
csv_file_path = './data/kinetics_400_labels.csv'

# Чтение меток классов из файла .csv
df = pd.read_csv(csv_file_path)
actions = df['action'].tolist()

# Загрузка списка опасных действий из отдельного .csv файла
dangerous_csv_file_path = './data/dangerous_actions.csv'
dangerous_df = pd.read_csv(dangerous_csv_file_path)
dangerous_actions = dangerous_df['action'].tolist()

logging.basicConfig(level=logging.DEBUG)

# Определение устройства (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.debug(f'Using device: {device}')

# Загрузка предобученной модели SlowFast
model = slowfast_r50(pretrained=True)
model = model.to(device).eval()

# Определение трансформации для входных кадров
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
])

# Инициализация детектора людей MediaPipe с использованием GPU
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Функция для подготовки slow и fast путей
def prepare_inputs(frames):
    slow_pathway = [transform(Image.fromarray(frame)) for frame in frames[::4]]  # slow pathway (берем каждый 4-й кадр)
    fast_pathway = [transform(Image.fromarray(frame)) for frame in frames]  # fast pathway (берем все кадры)
    return [torch.stack(slow_pathway).permute(1, 0, 2, 3).unsqueeze(0).to(device),
            torch.stack(fast_pathway).permute(1, 0, 2, 3).unsqueeze(0).to(device)]

# Функция для классификации действий с использованием модели
def classify_action(frames):
    inputs = prepare_inputs(frames)
    with torch.no_grad():
        outputs = model(inputs)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    max_prob, predicted = torch.max(probabilities, 1)
    predicted_index = predicted.item()
    confidence = max_prob.item()
    logging.debug(f"Predicted index: {predicted_index}, Confidence: {confidence}")

    if 0 <= predicted_index < len(actions):
        action = actions[predicted_index]
    else:
        action = "unknown"

    return action, confidence

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    camera_name = next((cam['name'] for cam in cameras if cam['id'] == camera_id), f"Camera {camera_id}")
    return Response(generate_frames_with_notification(camera_id, camera_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames_with_notification(camera_id, camera_name):
    cap = cv2.VideoCapture(camera_id)
    prev_time = 0
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break

        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Сбор кадров для распознавания действий
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) < 32:  # SlowFast требует минимум 32 кадра
            continue
        elif len(frames) > 32:
            frames.pop(0)

        # Обработка действий каждую секунду (или каждые 32 кадра)
        if len(frames) == 32 and fps >= 1:
            action, confidence = classify_action(frames)

            color = (0, 255, 0) if action not in dangerous_actions else (0, 0, 255)
            if action in dangerous_actions:
                notifications.send_telegram_notification(action, camera_name)
                with lock:
                    dangerous_actions_detected.append({'action': action, 'camera_name': camera_name})

            # Детекция человека и обводка прямоугольником с использованием MediaPipe
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                bbox = calculate_bounding_box(results.pose_landmarks.landmark, frame.shape)
                cv2.rectangle(frame, bbox[0], bbox[1], color, 2)

            cv2.putText(frame, f"Dangerous action detected: {action}" if action in dangerous_actions else f"Action: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75 if action in dangerous_actions else 0.5, color, 1, cv2.LINE_AA)

            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def calculate_bounding_box(landmarks, image_shape):
    image_height, image_width, _ = image_shape
    x_coords = [landmark.x * image_width for landmark in landmarks]
    y_coords = [landmark.y * image_height for landmark in landmarks]
    return ((int(min(x_coords)), int(min(y_coords))), (int(max(x_coords)), int(max(y_coords))))

@app.route('/get_dangerous_actions')
def get_dangerous_actions():
    with lock:
        actions = dangerous_actions_detected.copy()
        dangerous_actions_detected.clear()
    return jsonify(actions)

def start_camera_processing(camera):
    camera_id = camera['id']
    camera_name = camera['name']
    thread = Thread(target=video_feed, args=(camera_id,))
    thread.daemon = True
    thread.start()

def initialize_cameras():
    for camera in cameras:
        start_camera_processing(camera)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    initialize_cameras()
    app.run(host='0.0.0.0', port=5000, threaded=True)
