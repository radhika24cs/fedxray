"""
backend/xray/predict.py
Inference & Explainability Pipeline for Federated Chest X-ray Disease Detection.

Integrates:
- Loading saved Global Federated Model weights
- Preprocessing and tensor normalization
- Prediction with confidence and full class distribution
- Grad-CAM attention heatmap extraction
- Automated RAG medical information retrieval
"""

import os
import sys
import json
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.xray.model import FedChestNet, CLASSES
from backend.xray.preprocessing import normalize_intensity, TARGET_WIDTH, TARGET_HEIGHT
from backend.xray.dataset import decode_png
from backend.rag.retriever import MedicalRAGRetriever
from backend.rag.generator import MedicalRAGGenerator

class XrayPredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../saved_models/global_federated_model.json"))
        
        self.model = FedChestNet(seed=42)
        if os.path.exists(model_path):
            with open(model_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                self.model.set_weights(saved["weights"])
                self.metadata = saved
                print(f"[XrayPredictor] Loaded federated global model from {model_path}")
        else:
            print(f"[XrayPredictor] Warning: {model_path} not found. Using initialized weights.")
            self.metadata = {}

        self.rag_retriever = MedicalRAGRetriever()
        self.rag_generator = MedicalRAGGenerator(self.rag_retriever)

    def predict_image_file(self, image_path: str) -> Dict[str, Any]:
        """
        Runs complete analysis on a single X-ray image file.
        """
        width, height, raw_pixels = decode_png(image_path)
        return self.predict_pixels(raw_pixels, width, height)

    def predict_pixels(self, pixels: List[float], width: int = 224, height: int = 224) -> Dict[str, Any]:
        """
        Executes prediction, Grad-CAM generation, and RAG retrieval.
        """
        probs, cache = self.model.forward(pixels, width, height)

        # Ranked predictions
        ranked_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top_idx = ranked_indices[0]
        top_condition = CLASSES[top_idx]
        top_confidence = probs[top_idx]

        ranked_predictions = []
        for idx in ranked_indices:
            ranked_predictions.append({
                "condition": CLASSES[idx],
                "probability": round(probs[idx] * 100, 2),
                "is_top": idx == top_idx,
                "category": "Normal" if CLASSES[idx] == "Normal" else "Abnormal"
            })

        # Grad-CAM heatmap
        heatmap_28x28 = self.model.generate_gradcam(pixels, top_idx, cache)

        # Automated RAG retrieval
        rag_report = self.rag_generator.generate_condition_report(top_condition, top_confidence)

        return {
            "predicted_condition": top_condition,
            "confidence_percentage": round(top_confidence * 100, 2),
            "status": "Normal" if top_condition == "Normal" else "Abnormal Finding Detected",
            "ranked_predictions": ranked_predictions,
            "gradcam": {
                "grid_size": 28,
                "heatmap": heatmap_28x28,
                "label": "Model attention visualization — not a medical interpretation."
            },
            "rag_information": rag_report,
            "disclaimer": "AI research educational system — NOT a clinical diagnosis."
        }
