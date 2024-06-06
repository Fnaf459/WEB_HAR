document.addEventListener('DOMContentLoaded', (event) => {
    const alertContainer = document.getElementById('alert-container');
    let lastAlertTime = 0;
    const activeAlerts = new Set();

    function fetchDangerousActions() {
        const now = Date.now();
        if (now - lastAlertTime < 1000) {
            return;
        }

        fetch('/get_dangerous_actions')
            .then(response => response.json())
            .then(data => {
                data.forEach(action => {
                    if (!activeAlerts.has(action.camera_name)) {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = 'alert';
                        alertDiv.innerHTML = `Dangerous action detected: ${action.action} on camera ${action.camera_name}`;
                        alertContainer.appendChild(alertDiv);
                        activeAlerts.add(action.camera_name);

                        setTimeout(() => {
                            alertDiv.remove();
                            activeAlerts.delete(action.camera_name);
                        }, 5000);
                    }
                });
                lastAlertTime = now;
            })
            .catch(error => console.error('Error fetching dangerous actions:', error));
    }

    setInterval(fetchDangerousActions, 1000);
});

function showCameraMenu() {
    const menu = document.getElementById('camera-menu');
    menu.style.display = 'block';
    const button = document.getElementById('add-camera-button');
    menu.style.top = `${button.offsetTop + button.offsetHeight + 5}px`;
    menu.style.left = `${button.offsetLeft}px`;
}

function hideCameraMenu() {
    document.getElementById('camera-menu').style.display = 'none';
}

function addCamera(cameraId, cameraName) {
    fetch('/add_camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ camera_id: cameraId, camera_name: cameraName }),
    }).then(() => {
        window.location.reload();
    });
}

function showChangeCameraMenu(cameraId, button, cameraName) {
    const menu = document.getElementById('change-camera-menu');
    menu.style.display = 'block';
    menu.style.top = `${button.offsetTop}px`;
    const leftPos = button.offsetLeft + button.offsetWidth + 5;
    menu.style.left = `${leftPos}px`;
    document.getElementById('current-camera-id').value = cameraId;
    document.getElementById('current-camera-name').value = cameraName;
}

function hideChangeCameraMenu() {
    document.getElementById('change-camera-menu').style.display = 'none';
}

function changeCamera(newCameraId, newCameraName) {
    const currentCameraId = document.getElementById('current-camera-id').value;
    fetch('/change_camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ current_camera_id: currentCameraId, new_camera_id: newCameraId, new_camera_name: newCameraName }),
    }).then(() => {
        window.location.reload();
    });
}

function removeCamera(form) {
    fetch('/remove_camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams(new FormData(form)),
    }).then(() => {
        window.location.reload();
    });
    return false;
}

function setTelegramChatId(event) {
    event.preventDefault();
    const chatId = document.getElementById('chat_id').value;
    fetch('/set_telegram_chat_id', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({ chat_id: chatId })
    }).then(response => {
        if (response.status === 204) {
            alert('Notification chat ID set');
        } else {
            alert('Failed to set chat ID');
        }
    });
}

function updateCameraName(element) {
    const cameraId = element.getAttribute('data-camera-id');
    const newCameraName = element.innerText.trim();
    fetch('/change_camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ current_camera_id: cameraId, new_camera_id: cameraId, new_camera_name: newCameraName }),
    }).then(() => {
        alert('Camera name updated');
    });
}
