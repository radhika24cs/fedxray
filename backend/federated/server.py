"""
backend/federated/server.py
Federated Aggregation Server executing FedAvg across participating hospitals.

Functions:
- Coordinates communication rounds
- Aggregates updates via sample-weighted Federated Averaging
- Evaluates global model on independent multi-center validation & test sets
- Computes comprehensive IEEE evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
"""

import math
import json
import os
from typing import List, Dict, Any, Tuple
from backend.xray.model import FedChestNet, fedavg_aggregate, CLASSES
from backend.xray.dataset import ChestXrayDataset
from backend.federated.client import HospitalClient

class FederatedServer:
    def __init__(self, data_root: str):
        self.data_root = os.path.abspath(data_root)
        self.global_model = FedChestNet(seed=100)
        self.clients: List[HospitalClient] = []
        self.val_dataset = ChestXrayDataset(data_root=data_root, split="validation")
        self.test_dataset = ChestXrayDataset(data_root=data_root, split="test")
        self.round_history: List[Dict[str, Any]] = []

    def register_client(self, client: HospitalClient):
        self.clients.append(client)

    def evaluate_model(self, model: FedChestNet, dataset: ChestXrayDataset) -> Dict[str, Any]:
        """
        Evaluates a model instance against a designated dataset.
        Computes Accuracy, Macro Precision, Macro Recall, Macro F1, and Confusion Matrix.
        """
        num_classes = len(CLASSES)
        cm = [[0] * num_classes for _ in range(num_classes)]
        total = len(dataset)
        correct = 0
        total_loss = 0.0

        for i in range(total):
            item = dataset[i]
            probs, _ = model.forward(item["pixels"])
            label = item["label_idx"]

            eps = 1e-12
            loss = -math.log(max(eps, probs[label]))
            total_loss += loss

            pred = probs.index(max(probs))
            cm[label][pred] += 1
            if pred == label:
                correct += 1

        accuracy = correct / max(1, total)
        avg_loss = total_loss / max(1, total)

        # Compute per-class precision, recall, F1
        precisions = []
        recalls = []
        f1s = []

        for c in range(num_classes):
            tp = cm[c][c]
            fp = sum(cm[r][c] for r in range(num_classes) if r != c)
            fn = sum(cm[c][col] for col in range(num_classes) if col != c)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        macro_prec = sum(precisions) / num_classes
        macro_rec = sum(recalls) / num_classes
        macro_f1 = sum(f1s) / num_classes

        return {
            "loss": round(avg_loss, 4),
            "accuracy": round(accuracy * 100, 2),
            "precision": round(macro_prec * 100, 2),
            "recall": round(macro_rec * 100, 2),
            "f1_score": round(macro_f1 * 100, 2),
            "confusion_matrix": cm,
            "per_class": {
                CLASSES[c]: {
                    "precision": round(precisions[c] * 100, 2),
                    "recall": round(recalls[c] * 100, 2),
                    "f1": round(f1s[c] * 100, 2)
                } for c in range(num_classes)
            }
        }

    def run_round(self, round_num: int, local_epochs: int = 2) -> Dict[str, Any]:
        """
        Executes one complete Federated Learning communication cycle:
        1. Broadcast current global weights to Hospital A, B, and C
        2. Local private training on institutional data silos
        3. Collect weight updates
        4. Central FedAvg aggregation
        5. Update global model
        6. Validate on held-out multi-center test set
        """
        current_global_weights = self.global_model.get_weights()
        client_updates = []
        client_sample_counts = []
        client_metrics = []

        for client in self.clients:
            # Step 1: Send global model to hospital
            client.receive_global_model(current_global_weights)

            # Step 2: Hospital conducts private local training
            metrics = client.train_epoch(epochs=local_epochs)
            client_metrics.append({
                "hospital_id": client.hospital_id,
                "hospital_name": client.hospital_name,
                "local_loss": metrics["local_loss"],
                "local_accuracy": metrics["local_accuracy"],
                "samples": metrics["samples_trained"]
            })

            # Step 3: Hospital sends parameter update (NO images)
            client_updates.append(client.get_local_weights())
            client_sample_counts.append(client.num_samples)

        # Step 4: Central server performs FedAvg aggregation
        updated_weights = fedavg_aggregate(client_updates, client_sample_counts)
        self.global_model.set_weights(updated_weights)

        # Step 5: Evaluate on validation & global test sets
        val_eval = self.evaluate_model(self.global_model, self.val_dataset)
        test_eval = self.evaluate_model(self.global_model, self.test_dataset)

        round_summary = {
            "round": round_num,
            "clients": client_metrics,
            "validation_accuracy": val_eval["accuracy"],
            "validation_loss": val_eval["loss"],
            "global_test_accuracy": test_eval["accuracy"],
            "global_test_f1": test_eval["f1_score"],
            "global_test_loss": test_eval["loss"]
        }
        self.round_history.append(round_summary)
        return round_summary
