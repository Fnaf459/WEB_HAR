import requests
import logging
import time
import config

TELEGRAM_TOKEN = '7421576384:AAHYvYAOhyX01Taw23eckJJkVPsd6QHIe6Q'
last_notification_time = 0

def send_telegram_notification(action, camera_name):
    global last_notification_time

    current_time = time.time()
    if current_time - last_notification_time < 5:
        logging.debug("Skipping notification, too soon since last notification.")
        return

    if not config.notification_telegram_chat_id:
        logging.error("Chat ID is not set")
        return

    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {
        'chat_id': config.notification_telegram_chat_id,
        'text': f"Dangerous action detected: {action} on camera {camera_name}"
    }
    response = requests.post(url, json=payload)

    logging.debug(f"Telegram API response: {response.status_code} - {response.text}")

    last_notification_time = current_time

    return response
