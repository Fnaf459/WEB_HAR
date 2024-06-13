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

# Инициализация Flask приложения
app = Flask(__name__)
app.secret_key = 'supersecretkey'
register_routes(app)

# Глобальные переменные для хранения камер и обнаруженных опасных действий
cameras = []
lock = Lock()
dangerous_actions_detected = []

# Путь к файлам с метками классов
csv_file_path = './data/kinetics_400_labels_ru.csv'
dangerous_csv_file_path = './data/dangerous_actions_ru.csv'

# Функция для чтения CSV с автоматическим определением кодировки
def read_csv_with_fallback(file_path):
    try:
        # Определение кодировки
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        # Чтение файла с определенной кодировкой
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        # Альтернативное чтение с указанной кодировкой
        return pd.read_csv(file_path, encoding='latin1')

# Чтение меток классов и опасных действий с автоматическим определением кодировки
df = read_csv_with_fallback(csv_file_path)
dangerous_df = read_csv_with_fallback(dangerous_csv_file_path)

# Получение списка действий
actions = df['action'].tolist()
dangerous_actions = dangerous_df['action'].tolist()

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)

# Определение устройства (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.debug(f'Using device: {device}')

# Загрузка предобученной модели R3D-18 и экспорт в формат ONNX
class R3DWrapper(torch.nn.Module):
    def __init__(self, model):
        super(R3DWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        x = x.to(device)
        return self.model(x)

# Загрузка модели
model = r3d_18(pretrained=True)
wrapped_model = R3DWrapper(model).to(device).eval()

# # Функция для тонкой настройки модели на наборе данных Kinetics-400.
# # Параметры: модель, загрузчики данных для обучения и валидации, количество эпох обучения, скорость обучения.
# # Основные шаги: определение функции потерь и оптимизатора, цикл обучения с обновлением параметров модели, валидация после каждой эпохи.
# def fine_tune_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3):
#     criterion = torch.nn.CrossEntropyLoss() # Определение функции потерь
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, (inputs, labels) in enumerate(train_loader, 0): # Цикл по батчам данных
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward() # Обратное распространение ошибки
#             optimizer.step() # Обновление параметров модели
#             running_loss += loss.item()
#             if i % 10 == 9: # Печать информации о прогрессе каждые 10 батчей
#                 logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
#                 running_loss = 0.0
#         validate_model(model, val_loader)
#     logging.info('Finished fine-tuning')
#
# # Функция для валидации модели.
# # Параметры: модель, загрузчик данных для валидации.
# # Основные шаги: перевод модели в режим оценки, отключение вычисления градиентов, вычисление точности модели на валидационных данных.
# def validate_model(model, val_loader):
#     model.eval() # Перевод модели в режим оценки (валидации)
#     correct = 0
#     total = 0
#     with torch.no_grad(): # Отключение вычисления градиентов для ускорения
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1) # Определение предсказанных классов
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     logging.info(f'Accuracy of the network on the validation set: {100 * correct / total:.2f}%')
#
# # Функция загрузки данных Kinetics-400.
# # Параметры: размер батча, количество потоков для загрузки данных.
# # Основные шаги: определение преобразований для данных, загрузка обучающего и валидационного наборов данных, создание загрузчиков данных.
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
# # Создание загрузчиков данных
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#
#     return train_loader, val_loader
#
# # Example of how to use the fine-tuning function
# train_loader, val_loader = load_kinetics400_data()
# fine_tune_model(wrapped_model, train_loader, val_loader)

# Экспорт модели в формат ONNX
onnx_model_path = "./models/r3d_18.onnx"
dummy_input = torch.randn(1, 3, 8, 224, 224).to(device)  # R3D-18 принимает на вход 8 кадров
torch.onnx.export(wrapped_model, dummy_input, onnx_model_path, opset_version=11)

# Функция для создания движка TensorRT из ONNX модели
def build_engine(onnx_file_path, engine_file_path="./models/r3d_18.trt"):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # Создание билдера для компиляции сети
    builder = trt.Builder(TRT_LOGGER) # Создание сети с явной пакетной размерностью
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Чтение и парсинг ONNX файла
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors()):
                print(parser.get_error(error))
            return None

    # Установка лимита памяти и использование режима FP16 для ускорения
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    config.set_flag(trt.BuilderFlag.FP16)

    # Компиляция и сериализация сети в движок и сохранение сериализованного движка в файл
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine

# Оптимизация модели
trt_engine_path = "./models/r3d_18.trt"
engine = build_engine(onnx_model_path, trt_engine_path)

# Загрузка оптимизированного движка TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Определение преобразований для входных кадров
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
])

# Инициализация детектора людей с использованием MediaPipe с поддержкой GPU
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Функция для подготовки входных данных
def prepare_inputs(frames):
    transformed_frames = [transform(Image.fromarray(frame)) for frame in frames]
    return torch.stack(transformed_frames).unsqueeze(0).to(device)

# Функция для выделения буферов для входных и выходных данных
def allocate_buffers(engine, context):
    # Списки для хранения буферов входных и выходных данных
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream() # Создание потока CUDA для асинхронного выполнения операций

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        size = trt.volume(context.get_tensor_shape(tensor_name)) # Вычисляем объем памяти для тензора
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name)) # Определяем тип данных тензора
        host_mem = cuda.pagelocked_empty(size, dtype) # Выделяем память на хосте для тензора
        device_mem = cuda.mem_alloc(host_mem.nbytes) # Выделяем память на устройстве (GPU) для тензора
        bindings.append(int(device_mem))

        # Определяем, является ли тензор входным или выходным
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    return inputs, outputs, bindings, stream

# Функция для выполнения инференса
def do_inference(context, bindings, inputs, outputs, stream):
    # Перенос данных входа на GPU
    [cuda.memcpy_htod_async(inp["device"], inp["host"], stream) for inp in inputs]

    # Запуск инференса
    context.execute_v2(bindings=bindings)

    # Перенос результатов обратно с GPU
    [cuda.memcpy_dtoh_async(out["host"], out["device"], stream) for out in outputs]

    # Ожидание завершения работы GPU
    stream.synchronize()

    return [out["host"] for out in outputs]

# Подготовка буферов
inputs, outputs, bindings, stream = allocate_buffers(engine, context)

# Функция для классификации действия по кадрам
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

# Маршрут для потоковой передачи видео с камеры
@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    camera_name = next((cam['name'] for cam in cameras if cam['id'] == camera_id), f"Camera {camera_id}")
    return Response(generate_frames_with_notification(camera_id, camera_name), mimetype='multipart/x-mixed-replace; boundary=frame')

# Функция для генерации кадров с уведомлением об опасных действиях
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

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) < 8:
            continue
        elif len(frames) > 8:
            frames.pop(0)

        if len(frames) == 8 and fps >= 1:
            action, confidence = classify_action(frames)
            color = (0, 255, 0) if action not in dangerous_actions else (0, 0, 255)
            text_color = (0, 255, 0) if action not in dangerous_actions else (255, 0, 0)
            if action in dangerous_actions:
                notifications.send_telegram_notification(action, camera_name)
                with lock:
                    dangerous_actions_detected.append({'action': action, 'camera_name': camera_name})

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                bbox = calculate_bounding_box(results.pose_landmarks.landmark, frame.shape)
                cv2.rectangle(frame, bbox[0], bbox[1], color, 2)

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            small_font = ImageFont.truetype("arial.ttf", 12)
            text = f"Опасное действие: {action}" if action in dangerous_actions else f"Действие: {action}"
            draw.text((10, 60), text, font=small_font, fill=text_color)
            draw.text((10, 30), f'FPS: {fps}', font=small_font, fill=(0, 255, 0))
            draw.text((10, 90), f"Уверенность: {confidence:.2f}", font=small_font, fill=(0, 255, 0))

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

# Функция для вычисления ограничивающей рамки вокруг человека
def calculate_bounding_box(landmarks, image_shape):
    image_height, image_width, _ = image_shape
    x_coords = [landmark.x * image_width for landmark in landmarks]
    y_coords = [landmark.y * image_height for landmark in landmarks]
    return ((int(min(x_coords)), int(min(y_coords))), (int(max(x_coords)), int(max(y_coords))))

# Маршрут для получения списка обнаруженных опасных действий
@app.route('/get_dangerous_actions')
def get_dangerous_actions():
    with lock:
        actions = dangerous_actions_detected.copy()
        dangerous_actions_detected.clear()
    return jsonify(actions)

# Функция для запуска обработки камеры
def start_camera_processing(camera):
    camera_id = camera['id']
    camera_name = camera['name']
    thread = Thread(target=video_feed, args=(camera_id,))
    thread.daemon = True
    thread.start()

# Функция для инициализации камер
def initialize_cameras():
    for camera in cameras:
        start_camera_processing(camera)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    initialize_cameras()
    app.run(host='0.0.0.0', port=5000, threaded=True)
