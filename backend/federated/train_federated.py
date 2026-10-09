"""
backend/federated/train_federated.py
Master Federated Learning Training Orchestrator.

Implements:
1. Simulates 3 decentralized hospital institutions:
   - Hospital A (Pulmonology: High Pneumonia & Infiltration)
   - Hospital B (Cardiology: High Cardiomegaly & Pleural Effusion)
   - Hospital C (Metro General: High Atelectasis & Balanced)
2. Runs 6 Federated Rounds with FedAvg parameter aggregation.
3. Evaluates isolated single-hospital models to demonstrate the value of federation.
4. Serializes final global model to `backend/saved_models/global_federated_model.json`.
"""

import os
import sys
import glob
import math
import json
import random
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.xray.model import FedChestNet, extract_radiograph_features, softmax, CLASSES
from backend.xray.dataset import decode_png

def load_split_features(data_root: str, split_name: str) -> List[tuple]:
    """Pre-extracts radiograph features so federated training converges in seconds."""
    data = []
    base = os.path.join(data_root, split_name)
    for c_idx, c_name in enumerate(CLASSES):
        sub = c_name.lower().replace(" ", "_")
        folder = os.path.join(base, sub)
        if not os.path.exists(folder):
            continue
        for f in glob.glob(os.path.join(folder, "*.png")):
            w, h, p = decode_png(f)
            feats, _ = extract_radiograph_features(p, w, h)
            data.append((feats, c_idx, f))
    return data

def train_client_sgd(data, initial_weights, initial_bias, epochs=8, lr=0.025, weight_decay=0.001):
    num_classes = len(CLASSES)
    num_features = 64
    w = [list(row) for row in initial_weights]
    b = list(initial_bias)
    
    total_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
        shuffled = list(data)
        random.shuffle(shuffled)
        for feats, label, _ in shuffled:
            logits = [sum(feats[i] * w[c][i] for i in range(num_features)) + b[c] for c in range(num_classes)]
            probs = softmax(logits, temp=1.0)
            
            eps = 1e-12
            total_loss += -math.log(max(eps, probs[label]))
            if probs.index(max(probs)) == label:
                correct += 1
            total += 1

            for c in range(num_classes):
                target = 1.0 if c == label else 0.0
                grad = probs[c] - target
                for i in range(num_features):
                    w[c][i] -= lr * (grad * feats[i] + weight_decay * w[c][i])
                b[c] -= lr * grad

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return w, b, round(avg_loss, 4), round(acc, 4)

def evaluate_features(data, w, b, temp=1.2):
    num_classes = len(CLASSES)
    num_features = 64
    correct = 0
    total = len(data)
    total_loss = 0.0
    cm = [[0] * num_classes for _ in range(num_classes)]

    for feats, label, _ in data:
        logits = [sum(feats[i] * w[c][i] for i in range(num_features)) + b[c] for c in range(num_classes)]
        probs = softmax(logits, temp=temp)
        pred = probs.index(max(probs))
        cm[label][pred] += 1
        eps = 1e-12
        total_loss += -math.log(max(eps, probs[label]))
        if pred == label:
            correct += 1

    acc = (correct / total * 100.0) if total > 0 else 0.0
    avg_loss = (total_loss / total) if total > 0 else 0.0

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

    macro_prec = sum(precisions) / num_classes * 100.0
    macro_rec = sum(recalls) / num_classes * 100.0
    macro_f1 = sum(f1s) / num_classes * 100.0

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 2),
        "precision": round(macro_prec, 2),
        "recall": round(macro_rec, 2),
        "f1_score": round(macro_f1, 2),
        "confusion_matrix": cm,
        "per_class": {
            CLASSES[c]: {
                "precision": round(precisions[c] * 100.0, 1),
                "recall": round(recalls[c] * 100.0, 1),
                "f1": round(f1s[c] * 100.0, 1)
            } for c in range(num_classes)
        }
    }

def run_federated_training():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../saved_models"))
    os.makedirs(save_dir, exist_ok=True)

    print("======================================================================")
    print("STEP 8 & 9: MULTI-ROUND FEDERATED LEARNING EXPERIMENT (FedAvg)")
    print("Simulated Clinical Nodes: Hospital A, Hospital B, Hospital C")
    print("Zero Raw Patient Images Shared — Privacy Preserving Training")
    print("======================================================================")

    # 1. Load Pre-Extracted Features
    print("Loading radiograph dataset splits...")
    hosp_a_data = load_split_features(data_dir, "hospital_a")
    hosp_b_data = load_split_features(data_dir, "hospital_b")
    hosp_c_data = load_split_features(data_dir, "hospital_c")
    val_data    = load_split_features(data_dir, "validation")
    test_data   = load_split_features(data_dir, "test")

    print(f"Hospital A (Pulmonology)      : {len(hosp_a_data)} radiographs")
    print(f"Hospital B (Cardiovascular)   : {len(hosp_b_data)} radiographs")
    print(f"Hospital C (Metro General)    : {len(hosp_c_data)} radiographs")
    print(f"Validation Partition          : {len(val_data)} radiographs")
    print(f"Global Test Partition         : {len(test_data)} radiographs")

    num_classes = len(CLASSES)
    num_features = 64
    rng = random.Random(42)
    global_weights = [[rng.gauss(0, 0.05) for _ in range(num_features)] for _ in range(num_classes)]
    global_bias = [0.0 for _ in range(num_classes)]

    total_rounds = 5
    round_history = []
    total_train_samples = len(hosp_a_data) + len(hosp_b_data) + len(hosp_c_data)

    print("\n----------------------------------------------------------------------")
    print("COMMENCING FEDERATED ROUNDS...")
    print("----------------------------------------------------------------------")

    for r in range(1, total_rounds + 1):
        # 1. Local Hospital Client Training
        w_a, b_a, loss_a, acc_a = train_client_sgd(hosp_a_data, global_weights, global_bias, epochs=6, lr=0.025)
        w_b, b_b, loss_b, acc_b = train_client_sgd(hosp_b_data, global_weights, global_bias, epochs=6, lr=0.025)
        w_c, b_c, loss_c, acc_c = train_client_sgd(hosp_c_data, global_weights, global_bias, epochs=6, lr=0.025)

        # 2. Server-side Federated Averaging (FedAvg)
        n_a, n_b, n_c = len(hosp_a_data), len(hosp_b_data), len(hosp_c_data)
        for c in range(num_classes):
            for i in range(num_features):
                global_weights[c][i] = (n_a * w_a[c][i] + n_b * w_b[c][i] + n_c * w_c[c][i]) / total_train_samples
            global_bias[c] = (n_a * b_a[c] + n_b * b_b[c] + n_c * b_c[c]) / total_train_samples

        val_eval = evaluate_features(val_data, global_weights, global_bias)
        test_eval = evaluate_features(test_data, global_weights, global_bias)

        print(f"\n[FEDERATED ROUND {r}/{total_rounds}]")
        print(f"  ➜ Hospital A: Loss = {loss_a:.4f}, Acc = {acc_a*100:.1f}%")
        print(f"  ➜ Hospital B: Loss = {loss_b:.4f}, Acc = {acc_b*100:.1f}%")
        print(f"  ➜ Hospital C: Loss = {loss_c:.4f}, Acc = {acc_c*100:.1f}%")
        print(f"  ⚙ Server FedAvg Aggregated:")
        print(f"     Validation Accuracy : {val_eval['accuracy']:.2f}% | Loss: {val_eval['loss']:.4f}")
        print(f"     Global Test Accuracy: {test_eval['accuracy']:.2f}% | F1: {test_eval['f1_score']:.2f}%")

        round_history.append({
            "round": r,
            "validation_accuracy": val_eval["accuracy"],
            "validation_loss": val_eval["loss"],
            "global_test_accuracy": test_eval["accuracy"],
            "global_test_f1": test_eval["f1_score"],
            "clients": [
                {"hospital_name": "Hospital A", "local_loss": loss_a, "local_accuracy": acc_a},
                {"hospital_name": "Hospital B", "local_loss": loss_b, "local_accuracy": acc_b},
                {"hospital_name": "Hospital C", "local_loss": loss_c, "local_accuracy": acc_c},
            ]
        })

    # Final Comprehensive Global Test Evaluation
    final_eval = evaluate_features(test_data, global_weights, global_bias)
    print("\n----------------------------------------------------------------------")
    print(f"STEP 10: EVALUATING FINAL GLOBAL MODEL ON HELD-OUT TEST DATASET (N={len(test_data)})")
    print("----------------------------------------------------------------------")
    print(f"Final Global Test Accuracy : {final_eval['accuracy']:.2f}%")
    print(f"Final Global Macro Precision: {final_eval['precision']:.2f}%")
    print(f"Final Global Macro Recall   : {final_eval['recall']:.2f}%")
    print(f"Final Global Macro F1-Score : {final_eval['f1_score']:.2f}%")
    print("\nPer-Class Breakdown on Held-Out Test Radiographs:")
    for cls_name, met in final_eval["per_class"].items():
        print(f"  • {cls_name:<16}: Precision = {met['precision']:5.1f}%, Recall = {met['recall']:5.1f}%, F1 = {met['f1']:5.1f}%")

    # Benchmark Isolated Hospitals (No Federation)
    print("\n----------------------------------------------------------------------")
    print("STEP 7 COMPARISON: ISOLATED HOSPITAL MODELS (No Federation) VS GLOBAL")
    print("----------------------------------------------------------------------")
    isolated_results = {}
    hospitals = [
        ("hospital_a", "Hospital A (Pulmonology & Acute Care)", hosp_a_data),
        ("hospital_b", "Hospital B (Cardiovascular Center)", hosp_b_data),
        ("hospital_c", "Hospital C (Metro General Teaching)", hosp_c_data),
    ]
    for h_id, name, h_data in hospitals:
        w_iso, b_iso, _, _ = train_client_sgd(h_data, [[0.0]*64 for _ in range(6)], [0.0]*6, epochs=12, lr=0.03)
        iso_eval = evaluate_features(test_data, w_iso, b_iso)
        isolated_results[h_id] = {
            "name": name,
            "test_accuracy": iso_eval["accuracy"],
            "test_f1": iso_eval["f1_score"],
            "test_loss": iso_eval["loss"]
        }
        print(f"  • Isolated {name:<36} -> Test Accuracy: {iso_eval['accuracy']:.2f}%, F1: {iso_eval['f1_score']:.2f}%")

    print(f"  ★ Federated Global Model (FedAvg)           -> Test Accuracy: {final_eval['accuracy']:.2f}%, F1: {final_eval['f1_score']:.2f}%")

    # Save to disk
    model_save_path = os.path.join(save_dir, "global_federated_model.json")
    with open(model_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_architecture": "FedChestNet-6",
            "total_rounds": total_rounds,
            "num_classes": num_classes,
            "classes": CLASSES,
            "weights": {
                "fc_weights": global_weights,
                "fc_bias": global_bias,
                "num_features": num_features,
                "classes": CLASSES
            },
            "final_metrics": final_eval,
            "round_history": round_history
        }, f, indent=2)

    benchmark_path = os.path.join(save_dir, "benchmark_comparison.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump({
            "federated_global": final_eval,
            "isolated_hospitals": isolated_results,
            "round_history": round_history
        }, f, indent=2)

    print("\n======================================================================")
    print(f"STEP 11: Best Federated Model Saved to: {model_save_path}")
    print(f"Benchmark Comparison Saved to: {benchmark_path}")
    print("======================================================================")

if __name__ == "__main__":
    run_federated_training()
