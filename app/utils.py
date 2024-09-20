import threading
from datetime import datetime, timedelta
import base64
import numpy as np
from PIL import Image
import io

PING_TIMEOUT = timedelta(minutes=10)
shutdown_timers = {}
last_ping_times = {}

def reset_shutdown_timer(client_id, model_instances):
    global last_ping_times, shutdown_timers

    last_ping_times[client_id] = datetime.now()

    if client_id in shutdown_timers:
        shutdown_timers[client_id].cancel()

    shutdown_timers[client_id] = threading.Timer(PING_TIMEOUT.total_seconds(), shutdown_instance, args=[client_id, model_instances])
    shutdown_timers[client_id].start()

def shutdown_instance(client_id, model_instances):
    global shutdown_timers, last_ping_times
    if client_id in model_instances:
        del model_instances[client_id]
        del shutdown_timers[client_id]
        del last_ping_times[client_id]

def decode_base64_image(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)
    return image_np