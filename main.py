import base64
import io
import torch
import logging
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import uuid
from sam2.build_sam import build_sam2
from flask_cors import CORS 
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt

# Initialiser Flask
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], methods=['GET', 'POST'], allow_headers=['Content-Type'])

# Configurer le logger
logging.basicConfig(level=logging.INFO)

# Détection de l'appareil (GPU, MPS ou CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logging.warning("MPS support is preliminary and may cause issues.")
else:
    device = torch.device("cpu")

logging.info(f"using device: {device}")

# Charger le modèle SAM 2
checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

try:
    sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    logging.info("Modèle SAM 2 chargé avec succès")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}")
    raise

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

@app.route('/predict', methods=['POST'])
def predict_masks():
    try:
        # Vérifier que l'image est présente dans la requête
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'Image non fournie'}), 400

        # Décoder l'image en base64
        image_base64 = data['image']
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        # Générer les masques avec SAM 2
        masks = mask_generator.generate(image_np)

        # Convertir les masques en format sérialisable
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

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction des masques : {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)