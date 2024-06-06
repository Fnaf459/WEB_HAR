from flask import request, render_template, jsonify, Response, flash, get_flashed_messages
from camera import get_available_cameras, generate_frames
from notifications import send_telegram_notification
from telegram import Bot
import logging
import config

bot = Bot(token='7421576384:AAHYvYAOhyX01Taw23eckJJkVPsd6QHIe6Q')


def register_routes(app):
    @app.route('/')
    def index():
        available_cameras = get_available_cameras()
        return render_template('index.html', cameras=config.cameras, available_cameras=available_cameras,
                               notifications=get_flashed_messages())

    @app.route('/add_camera', methods=['POST'])
    def add_camera():
        data = request.json
        camera_id = int(data['camera_id'])
        camera_name = data.get('camera_name', f"Camera {camera_id}")

        if not any(cam['id'] == camera_id for cam in config.cameras):
            config.cameras.append({'id': camera_id, 'name': camera_name})
            return ('', 204)
        else:
            return jsonify({'error': 'Camera already exists'}), 400

    @app.route('/remove_camera', methods=['POST'])
    def remove_camera():
        camera_id = int(request.form['camera_id'])
        config.cameras = [cam for cam in config.cameras if cam['id'] != camera_id]
        return ('', 204)

    @app.route('/change_camera', methods=['POST'])
    def change_camera():
        data = request.json
        current_camera_id = int(data['current_camera_id'])
        new_camera_id = int(data['new_camera_id'])
        new_camera_name = data.get('new_camera_name', f"Camera {new_camera_id}")

        for cam in config.cameras:
            if cam['id'] == current_camera_id:
                cam['id'] = new_camera_id
                cam['name'] = new_camera_name
                break
        return ('', 204)

    @app.route('/set_telegram_chat_id', methods=['POST'])
    def set_telegram_chat_id():
        config.notification_telegram_chat_id = request.form['chat_id']
        logging.debug(f"Chat ID set to: {config.notification_telegram_chat_id}")
        return ('', 204)
