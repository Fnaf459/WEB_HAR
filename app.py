import logging
import time
from threading import Thread, Lock

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import chardet
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw
from flask import Flask, Response, jsonify
from torchvision.models.video import r3d_18
from torchvision.datasets.kinetics import Kinetics
from torch.utils.data import DataLoader

import notifications
from routes import register_routes

app = Flask(__name__)
app.secret_key = 'supersecretkey'
register_routes(app)

cameras = []
lock = Lock()
dangerous_actions_detected = []

# Path to the file with class labels
csv_file_path = './data/kinetics_400_labels_ru.csv'
dangerous_csv_file_path = './data/dangerous_actions_ru.csv'

def read_csv_with_fallback(file_path):
    try:
        # Detect encoding
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        # Try reading with detected encoding
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        # Fallback to a specified encoding
        return pd.read_csv(file_path, encoding='latin1')

# Read class labels and dangerous actions with encoding fallback
df = read_csv_with_fallback(csv_file_path)
dangerous_df = read_csv_with_fallback(dangerous_csv_file_path)

# Get action lists
actions = df['action'].tolist()
dangerous_actions = dangerous_df['action'].tolist()

logging.basicConfig(level=logging.DEBUG)

# Define the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.debug(f'Using device: {device}')

# Load the pre-trained R3D-18 model and export it to ONNX
class R3DWrapper(torch.nn.Module):
    def __init__(self, model):
        super(R3DWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        x = x.to(device)
        return self.model(x)

model = r3d_18(pretrained=True)
wrapped_model = R3DWrapper(model).to(device).eval()

# # Function for fine-tuning the model on Kinetics-400 dataset
# def fine_tune_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3):
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, (inputs, labels) in enumerate(train_loader, 0):
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             if i % 10 == 9:
#                 logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
#                 running_loss = 0.0
#         validate_model(model, val_loader)
#     logging.info('Finished fine-tuning')
#
# # Function for validating the model
# def validate_model(model, val_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     logging.info(f'Accuracy of the network on the validation set: {100 * correct / total:.2f}%')
#
# # Load Kinetics-400 dataset
# def load_kinetics400_data(batch_size=8, num_workers=4):
#     data_transform = transforms.Compose([
#         transforms.Resize((128, 171)),
#         transforms.CenterCrop(112),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
#     ])
#
#     train_dataset = Kinetics(
#         root='./data',
#         frames_per_clip=16,
#         num_classes='400',
#         split='train',
#         transform=data_transform,
#         download=True,
#         num_workers=num_workers
#     )
#     val_dataset = Kinetics(
#         root='./data',
#         frames_per_clip=16,
#         num_classes='400',
#         split='val',
#         transform=data_transform,
#         download=True,
#         num_workers=num_workers
#     )
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#
#     return train_loader, val_loader
#
# # Example of how to use the fine-tuning function
# train_loader, val_loader = load_kinetics400_data()
# fine_tune_model(wrapped_model, train_loader, val_loader)

# Export the model to ONNX format
onnx_model_path = "./models/r3d_18.onnx"
dummy_input = torch.randn(1, 3, 8, 224, 224).to(device)  # R3D-18 takes a single input of 8 frames
torch.onnx.export(wrapped_model, dummy_input, onnx_model_path, opset_version=11)

# Function to build the TensorRT engine
def build_engine(onnx_file_path, engine_file_path="./models/r3d_18.trt"):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors()):
                print(parser.get_error(error))
            return None

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine

# Optimize the model
trt_engine_path = "./models/r3d_18.trt"
engine = build_engine(onnx_model_path, trt_engine_path)

# Load the optimized TensorRT model
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Define the transformation for input frames
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
])

# Initialize the MediaPipe human detector with GPU support
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, static_image_mode=False, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Function to prepare input
def prepare_inputs(frames):
    transformed_frames = [transform(Image.fromarray(frame)) for frame in frames]
    return torch.stack(transformed_frames).unsqueeze(0).to(device)

def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        size = trt.volume(context.get_tensor_shape(tensor_name))
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    # Переносим данные входа на GPU
    [cuda.memcpy_htod_async(inp["device"], inp["host"], stream) for inp in inputs]

    # Запуск инференса
    context.execute_v2(bindings=bindings)

    # Переносим результаты обратно с GPU
    [cuda.memcpy_dtoh_async(out["host"], out["device"], stream) for out in outputs]

    # Ожидание завершения работы GPU
    stream.synchronize()

    return [out["host"] for out in outputs]

# Prepare buffers
inputs, outputs, bindings, stream = allocate_buffers(engine, context)

def classify_action(frames):
    inputs_data = prepare_inputs(frames)
    np.copyto(inputs[0]["host"], inputs_data.cpu().numpy().ravel())

    outputs_data = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    probabilities = torch.tensor(outputs_data[0]).reshape(1, -1)
    probabilities = F.softmax(probabilities, dim=1)
    max_prob, predicted = torch.max(probabilities, 1)
    predicted_index = predicted.item()
    confidence = max_prob.item()

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
    if not cap.isOpened():
        logging.error(f"Could not open video device {camera_id}")
        return

    prev_time = 0
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            logging.error(f"Failed to capture image from camera {camera_id}")
            break

        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Collect frames for action recognition
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) < 8:  # R3D-18 requires at least 8 frames
            continue
        elif len(frames) > 8:
            frames.pop(0)

        # Process actions every second (or every 8 frames)
        if len(frames) == 8 and fps >= 1:
            action, confidence = classify_action(frames)

            color = (0, 255, 0) if action not in dangerous_actions else (0, 0, 255)  # Green for normal, red for dangerous
            text_color = (0, 255, 0) if action not in dangerous_actions else (255, 0, 0)
            if action in dangerous_actions:
                notifications.send_telegram_notification(action, camera_name)
                with lock:
                    dangerous_actions_detected.append({'action': action, 'camera_name': camera_name})

            # Detect humans and draw bounding boxes using MediaPipe
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                bbox = calculate_bounding_box(results.pose_landmarks.landmark, frame.shape)
                cv2.rectangle(frame, bbox[0], bbox[1], color, 2)

            # Convert frame to PIL for text drawing with Cyrillic support
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            small_font = ImageFont.truetype("arial.ttf", 12)  # Half size font

            text = f"Опасное действие: {action}" if action in dangerous_actions else f"Действие: {action}"
            draw.text((10, 60), text, font=small_font, fill=text_color)
            draw.text((10, 30), f'FPS: {fps}', font=small_font, fill=(0, 255, 0))
            draw.text((10, 90), f"Уверенность: {confidence:.2f}", font=small_font, fill=(0, 255, 0))

            # Convert back to OpenCV frame
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Failed to encode frame")
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    logging.info(f"Released camera {camera_id}")

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
