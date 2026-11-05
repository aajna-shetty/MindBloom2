"""
app/video/emotion_processor.py
Vision Transformer (ViT) based facial expression recognition processor.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


class EmotionProcessor:
    """
    Analyzes facial expressions using a Vision Transformer (ViT) model.
    Emits 7 standard normalized emotion scores.
    """
    
    STANDARD_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    LABEL_SYNONYMS = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "disgusted": "disgust",
        "fear": "fear",
        "fearful": "fear",
        "happy": "happy",
        "happiness": "happy",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
        "surprised": "surprise",
        "neutral": "neutral"
    }

    def __init__(self, model_id: str = "mo-thecreator/vit-Facial-Expression-Recognition"):
        self.model_id = model_id
        self.device = torch.device("cpu")
        
        torch.set_num_threads(max(1, torch.get_num_threads() // 2))
        
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        
        self.idx_to_standard = {}
        for raw_idx, raw_label in self.model.config.id2label.items():
            clean_label = raw_label.strip().lower()
            standard_label = self.LABEL_SYNONYMS.get(clean_label, clean_label)
            self.idx_to_standard[int(raw_idx)] = standard_label
            
        print(f"[EmotionProcessor] Loaded id2label mapping: {self.idx_to_standard}")
        self._warmup()

    def _warmup(self) -> None:
        dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        inputs = self.processor(images=dummy_img, return_tensors="pt")
        with torch.inference_mode():
            _ = self.model(**inputs)

    def analyze(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Runs ViT emotion classification on a cropped face image using the
        original model logits and standard softmax.
        """
        if face_image is None or face_image.size == 0:
            return {
                "success": False,
                "dominant_emotion": "neutral",
                "emotion_scores": {e: 0.0 for e in self.STANDARD_EMOTIONS},
                "error": "Empty or invalid face frame"
            }

        try:
            rgb_crop = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_crop)
            
            inputs = self.processor(images=pil_img, return_tensors="pt")
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits[0].clone()

                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

            emotion_scores = {e: 0.0 for e in self.STANDARD_EMOTIONS}
            for idx, prob in enumerate(probs):
                std_key = self.idx_to_standard.get(idx)
                if std_key in emotion_scores:
                    emotion_scores[std_key] = float(prob)

            dominant = max(emotion_scores, key=emotion_scores.get)

            return {
                "success": True,
                "dominant_emotion": dominant,
                "emotion_scores": emotion_scores,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "dominant_emotion": "neutral",
                "emotion_scores": {e: 0.0 for e in self.STANDARD_EMOTIONS},
                "error": str(e)
            }