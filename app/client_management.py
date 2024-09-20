from .model import load_model, predict
from .utils import reset_shutdown_timer, decode_base64_image
import uuid
import numpy as np

model_instances = {}
last_ping_times = {}
shutdown_timers = {}

def instantiate_model(client_id):
    if client_id not in model_instances:
        model, mask_generator = load_model() 
        model_instances[client_id] = {'model': model, 'mask_generator': mask_generator}
        reset_shutdown_timer(client_id, model_instances)
        return {"status": f"Model instantiated for client {client_id}"}
    else:
        return {"status": f"Model already instantiated for client {client_id}"}

def handle_ping(client_id):
    if client_id in model_instances:
        reset_shutdown_timer(client_id, model_instances)
        return {"status": f"Ping received, model instance for client {client_id} kept alive"}
    else:
        return {"status": f"No model instance active for client {client_id}"}

def predict_for_client(client_id, image_base64):
    if client_id not in model_instances:
        return {"error": f"Model not instantiated for client {client_id}"}

    image_np = decode_base64_image(image_base64)
    model_instance = model_instances[client_id]
    masks = predict(model_instance, image_np)
    
    masks_serialized = []
    for mask in masks:
        masks_serialized.append({
            'segmentation': mask['segmentation'].tolist() if isinstance(mask['segmentation'], np.ndarray) else mask['segmentation'],
            'area': mask['area'],
            'bbox': mask['bbox'],
            'predicted_iou': mask['predicted_iou'],
            'point_coords': mask['point_coords'].tolist() if isinstance(mask['point_coords'], np.ndarray) else mask['point_coords'],
            'stability_score': mask['stability_score'],
            'crop_box': mask['crop_box']
        })

    # Générer un identifiant unique pour cette image
    image_id = str(uuid.uuid4())

    # Créer la réponse JSON
    response = {
        'image_id': image_id,
        'masks': masks_serialized
    }
    
    reset_shutdown_timer(client_id, model_instances)
    return {"status": "Prediction completed", "masks": response}