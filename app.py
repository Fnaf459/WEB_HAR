from flask import Flask, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import requests
import time
from routes import register_routes
from threading import Thread, Lock
import logging
import notifications

app = Flask(__name__)
app.secret_key = 'supersecretkey'
register_routes(app)

cameras = []
lock = Lock()
dangerous_actions_detected = []

mp_pose = mp.solutions.pose

actions = ["standing", "walking", "running", "jumping", "sitting", "lying", "punching"]
dangerous_actions = ["running", "jumping", "punching", "standing"]

# Загрузка модели
MODEL_URL = "https://example.com/path/to/your/model.tflite"
MODEL_PATH = "path/to/model/model.tflite"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)

logging.basicConfig(level=logging.DEBUG)

def classify_pose(landmarks):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    if (left_hip.visibility < 0.5 or right_hip.visibility < 0.5 or
        left_knee.visibility < 0.5 or right_knee.visibility < 0.5 or
        left_ankle.visibility < 0.5 or right_ankle.visibility < 0.5 or
        left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5):
        return "unknown"

    def calculate_angle(a, b, c):
        angle = np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x)
        return np.abs(angle * 180.0 / np.pi)

    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    if left_leg_angle > 160 and right_leg_angle > 160:
        return "standing"

    if (left_leg_angle > 150 and right_leg_angle < 160) or (right_leg_angle > 150 and left_leg_angle < 160):
        return "walking"

    if (left_leg_angle < 140 and right_leg_angle < 140) and (left_ankle.y < left_hip.y or right_ankle.y < right_hip.y):
        return "running"

    if (left_ankle.y < left_hip.y and right_ankle.y < right_hip.y) and (left_leg_angle < 160 or right_leg_angle < 160):
        return "jumping"

    if (left_leg_angle < 100 and right_leg_angle < 100) and (left_hip.y > left_knee.y and right_hip.y > right_knee.y):
        return "sitting"

    if left_hip.y > left_shoulder.y and right_hip.y > right_shoulder.y:
        return "lying"

    if ((left_arm_angle < 45 or right_arm_angle < 45) and (left_arm_angle > 120 or right_arm_angle > 120)):
        return "punching"

    return "unknown"

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    camera_name = next((cam['name'] for cam in cameras if cam['id'] == camera_id), f"Camera {camera_id}")
    return Response(generate_frames_with_notification(camera_id, camera_name), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames_with_notification(camera_id, camera_name):
    cap = cv2.VideoCapture(camera_id)
    pose = mp_pose.Pose()
    prev_time = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        action = "unknown"
        confidence = 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            action = classify_pose(landmarks)
            color = (0, 255, 0) if action not in dangerous_actions else (0, 0, 255)
            confidence = 0.95 if action in dangerous_actions else 0.9

            if action in dangerous_actions:
                notifications.send_telegram_notification(action, camera_name)
                cv2.putText(frame, f"Dangerous action detected: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
                with lock:
                    dangerous_actions_detected.append({'action': action, 'camera_name': camera_name})
            else:
                cv2.putText(frame, f"Action: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, color, -1)

            min_x = int(min(landmark.x for landmark in landmarks) * frame.shape[1])
            max_x = int(max(landmark.x for landmark in landmarks) * frame.shape[1])
            min_y = int(min(landmark.y for landmark in landmarks) * frame.shape[0])
            max_y = int(max(landmark.y for landmark in landmarks) * frame.shape[0])
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)

        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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
