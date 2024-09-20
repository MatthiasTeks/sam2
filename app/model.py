import torch
import logging
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

model_instances = {}

def load_model():
    checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"
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

    model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(model)
    return model, mask_generator

def predict(model_instance, image_np):
    # Générer les masques
    logging.info("1")
    mask_generator = model_instance["mask_generator"]
    logging.info("2")
    masks = mask_generator.generate(image_np)

    logging.info("Prédiction effectuée")
    return masks