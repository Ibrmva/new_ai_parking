import torch
import cv2
import numpy as np
import os
import sys
import logging
from strhub.models.parseq.system import PARSeq
import yaml
from PIL import Image
import base64

logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class MLService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MLService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.device = torch.device(os.getenv('ML_DEVICE', 'mps'))
        self.lpd_model = None
        self.lpr_model = None
        self._load_models()

    def _load_models(self):
        logger.info(f"ML device set to: {self.device}")

        lpd_path = os.getenv('LPD_MODEL_PATH', 'lpr/models/lpd/license_plate_detector.pt')
        logger.info(f"LPD model path: {lpd_path}")
        if os.path.exists(lpd_path):
            try:
                from ultralytics import YOLO
                logger.info("Loading LPD model...")
                self.lpd_model = YOLO(lpd_path)
                self.lpd_model.to(self.device)
                logger.info("LPD model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LPD model: {e}")
                self.lpd_model = None
        else:
            logger.error(f"LPD model file not found at {lpd_path}")
            self.lpd_model = None

        lpr_path = os.getenv('LPR_MODEL_PATH', 'lpr/models/lpr/evaluated_ocr_model.pt')
        logger.info(f"LPR model path: {lpr_path}")
        if os.path.exists(lpr_path):
            try:
                logger.info("Loading LPR model...")
                checkpoint = torch.load(lpr_path, map_location=self.device, weights_only=False)
                config_path = 'lpr/configs/model/parseq.yaml'
                charset_path = 'lpr/configs/charset/label.yaml'

                if not os.path.exists(config_path):
                    logger.error(f"Config file not found: {config_path}")
                    self.lpr_model = None
                    return

                if not os.path.exists(charset_path):
                    logger.error(f"Charset file not found: {charset_path}")
                    self.lpr_model = None
                    return

                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                with open(charset_path, 'r') as f:
                    charset_config = yaml.safe_load(f)

                self.lpr_model = PARSeq(
                    charset_train=charset_config['model']['charset_train'],
                    charset_test=charset_config['model']['charset_train'],
                    max_label_length=config.get('max_label_length', 25),
                    batch_size=config.get('batch_size', 64),
                    lr=config.get('lr', 7e-4),
                    warmup_pct=config.get('warmup_pct', 0.075),
                    weight_decay=config.get('weight_decay', 0.0),
                    img_size=config.get('img_size', [32, 128]),
                    patch_size=config.get('patch_size', [4, 8]),
                    embed_dim=config.get('embed_dim', 384),
                    enc_num_heads=config.get('enc_num_heads', 6),
                    enc_mlp_ratio=config.get('enc_mlp_ratio', 4),
                    enc_depth=config.get('enc_depth', 12),
                    dec_num_heads=config.get('dec_num_heads', 12),
                    dec_mlp_ratio=config.get('dec_mlp_ratio', 4),
                    dec_depth=config.get('dec_depth', 1),
                    perm_num=config.get('perm_num', 6),
                    perm_forward=config.get('perm_forward', True),
                    perm_mirrored=config.get('perm_mirrored', True),
                    decode_ar=config.get('decode_ar', True),
                    refine_iters=config.get('refine_iters', 1),
                    dropout=config.get('dropout', 0.1)
                )
                self.lpr_model.load_state_dict(checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint)
                self.lpr_model.to(self.device)
                self.lpr_model.eval()
                logger.info("LPR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LPR model: {e}")
                self.lpr_model = None
        else:
            logger.error(f"LPR model file not found at {lpr_path}")
            self.lpr_model = None
 
    def detect_plates(self, image_source, original_image_bytes=None):
        cv_image = None
        original_format = 'JPEG'
        if isinstance(image_source, str):
            pil_image = Image.open(image_source)
            original_format = pil_image.format or 'JPEG'
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            cv_image = cv2.imread(image_source)
            if cv_image is None and original_image_bytes:
                np_img = np.frombuffer(original_image_bytes, np.uint8)
                cv_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        elif isinstance(image_source, np.ndarray):
            cv_image = image_source

        if cv_image is None:
            logger.error("Failed to load image for plate detection")
            return []

        h_img, w_img = cv_image.shape[:2]
        logger.info(f"Processing image with dimensions: {w_img}x{h_img}")
        margin = 10
        plates = []

        try:
            if self.lpd_model:
                logger.info("Using YOLO model for plate detection")
                results = self.lpd_model(cv_image)
                logger.info(f"YOLO model returned {len(results)} results")

                for result in results:
                    logger.info(f"Result has {len(result.boxes)} boxes")
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        logger.info(f"Detected box with confidence {conf:.3f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

                        x1 = max(0, int(x1)-margin)
                        y1 = max(0, int(y1)-margin)
                        x2 = min(w_img, int(x2)+margin)
                        y2 = min(h_img, int(y2)+margin)
                        cropped = cv_image[y1:y2, x1:x2]
                        plates.append({
                            "confidence": float(conf),
                            "cropped_image": cropped,
                            "original_bytes": original_image_bytes,
                            "original_format": original_format,
                            "bbox": [x1, y1, x2, y2]
                        })
                logger.info(f"Total plates detected: {len(plates)}")
            else:
                logger.warning("LPD model not loaded, using fallback detection")
                x1, y1, x2, y2 = int(w_img*0.1), int(h_img*0.1), int(w_img*0.8), int(h_img*0.8)
                cropped = cv_image[y1:y2, x1:x2]
                plates.append({
                    "confidence": 0.95,
                    "cropped_image": cropped,
                    "original_bytes": original_image_bytes,
                    "original_format": original_format,
                    "bbox": [x1, y1, x2, y2]
                })
        except Exception as e:
            logger.error(f"Plate detection error: {e}")
            logger.error(f"Error details: {str(e)}", exc_info=True)

        return plates

    def recognize(self, model_type, cropped_image):
        if self.lpr_model is None:
            return "", ([], [])
        corrected = self._correct_rotation(cropped_image)
        resized = cv2.resize(corrected, (128, 32))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.lpr_model(tensor, max_length=12)
            tokens, probs = self.lpr_model.tokenizer.decode(logits.softmax(-1))
            recognized_text = tokens[0]
            confidences = [f"{p:.3f}" for p in probs[0].tolist()]
        return recognized_text, (list(recognized_text), confidences)

    def _correct_rotation(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        angle = rect[2]
        if angle < -45: angle += 90
        elif angle > 45: angle -= 90
        if abs(angle) > 5:
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
            return cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return image

ml_service = MLService()
