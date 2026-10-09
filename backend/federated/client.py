"""
backend/federated/client.py
Simulated Hospital Client for Federated Learning.

Privacy Principles:
- Patient radiographs are kept strictly inside the hospital's local silo.
- Only mathematical model parameters / weight deltas are transmitted to the coordinator.
- Raw medical images NEVER leave the institutional boundary.
"""

import math
import random
from typing import Dict, Any, Tuple, List
from backend.xray.model import FedChestNet, CLASSES
from backend.xray.dataset import ChestXrayDataset, ChestXrayDataLoader

class HospitalClient:
    def __init__(self, hospital_id: str, hospital_name: str, data_root: str, split_name: str, lr: float = 0.025):
        self.hospital_id = hospital_id
        self.hospital_name = hospital_name
        self.split_name = split_name
        self.lr = lr
        self.dataset = ChestXrayDataset(data_root=data_root, split=split_name)
        self.num_samples = len(self.dataset)
        self.local_model = FedChestNet(seed=42)

    def receive_global_model(self, global_weights: Dict[str, Any]):
        """Downloads the current global model parameter state."""
        self.local_model.set_weights(global_weights)

    def train_epoch(self, epochs: int = 6, batch_size: int = 16, weight_decay: float = 0.001) -> Dict[str, float]:
        """
        Executes local SGD on the private hospital dataset.
        Updates model weights locally without sending any patient data outward.
        """
        loader = ChestXrayDataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            for batch in loader:
                for pixels, label_idx, one_hot in zip(batch["pixels"], batch["labels"], batch["one_hot"]):
                    probs, cache = self.local_model.forward(pixels)
                    
                    # Cross-Entropy Loss
                    eps = 1e-12
                    loss = -math.log(max(eps, probs[label_idx]))
                    total_loss += loss
                    
                    pred_idx = probs.index(max(probs))
                    if pred_idx == label_idx:
                        correct += 1
                    total += 1

                    # Gradient step on classifier weights
                    feats = cache["features"]
                    for c in range(self.local_model.num_classes):
                        grad_out = probs[c] - one_hot[c]
                        for i in range(self.local_model.num_features):
                            self.local_model.fc_weights[c][i] -= self.lr * (grad_out * feats[i] + weight_decay * self.local_model.fc_weights[c][i])
                        self.local_model.fc_bias[c] -= self.lr * grad_out

        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        return {
            "local_loss": round(avg_loss, 4),
            "local_accuracy": round(accuracy, 4),
            "samples_trained": self.num_samples
        }

    def get_local_weights(self) -> Dict[str, Any]:
        """Exports local model parameters to send to the central aggregation server."""
        return self.local_model.get_weights()
