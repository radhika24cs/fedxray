import sys
sys.path.insert(0, ".")
import os, glob, math, json, random
from backend.xray.dataset import decode_png

CLASSES = [
    "Normal",
    "Pneumonia",
    "Atelectasis",
    "Cardiomegaly",
    "Pleural Effusion",
    "Infiltration"
]

def extract_radiograph_features(pixels, width=224, height=224):
    grid = [pixels[r * width:(r + 1) * width] for r in range(height)]
    features = []

    # 1. 16 Spatial Grid Block Means (4x4)
    bh = height // 4
    bw = width // 4
    block_means = []
    block_stds = []

    for br in range(4):
        for bc in range(4):
            pts = [grid[r][c] for r in range(br * bh, (br + 1) * bh) for c in range(bc * bw, (bc + 1) * bw)]
            m = sum(pts) / len(pts)
            var = sum((p - m) ** 2 for p in pts) / len(pts)
            s = math.sqrt(var)
            block_means.append(m)
            block_stds.append(s)
            features.append((m - 98.0) / 35.0)

    # 2. 16 Spatial Grid Block STDs (4x4)
    for s in block_stds:
        features.append((s - 32.0) / 18.0)

    # 3. 8 Bilateral Hemithorax Asymmetry Indices (Left vs Right per tier)
    for tier in range(4):
        right_side = (block_means[tier * 4 + 0] + block_means[tier * 4 + 1]) / 2.0
        left_side  = (block_means[tier * 4 + 2] + block_means[tier * 4 + 3]) / 2.0
        diff = (left_side - right_side) / 25.0
        features.append(diff)
        features.append(abs(diff))

    # 4. 8 Anatomical Landmark Indices
    # Cardiac region: Tier 2 (rows 112-168), cols 1 and 2
    cardiac_val = (block_means[2 * 4 + 1] + block_means[2 * 4 + 2]) / 2.0
    thoracic_border = (block_means[2 * 4 + 0] + block_means[2 * 4 + 3]) / 2.0
    ctr_proxy = (cardiac_val - thoracic_border) / 30.0
    features.append(ctr_proxy)

    # Left cardiac apex prominence (Tier 2, col 2 and 3)
    left_apex = (block_means[2 * 4 + 2] - block_means[1 * 4 + 2]) / 25.0
    features.append(left_apex)

    # Costophrenic sulci (Tier 3: col 0 is right, col 3 is left)
    right_cp = (block_means[3 * 4 + 0] - 85.0) / 25.0
    left_cp  = (block_means[3 * 4 + 3] - 85.0) / 25.0
    features.append(left_cp)
    features.append(right_cp)
    features.append(min(left_cp, right_cp))

    # Perihilar zone (Tier 1 & 2 central: col 1 & 2)
    perihilar = (block_means[1 * 4 + 1] + block_means[1 * 4 + 2]) / 2.0
    features.append((perihilar - 95.0) / 25.0)

    # Global lung aeration (mean of upper & mid lateral zones)
    aeration = (block_means[0 * 4 + 0] + block_means[0 * 4 + 3] + block_means[1 * 4 + 0] + block_means[1 * 4 + 3]) / 4.0
    features.append((aeration - 100.0) / 30.0)

    # Basal-to-apical gradient (Tier 3 vs Tier 0)
    basal_mean = sum(block_means[12:16]) / 4.0
    apical_mean = sum(block_means[0:4]) / 4.0
    features.append((basal_mean - apical_mean) / 30.0)

    # 5. 16 Multi-Frequency Gradient & Edge Profiling Features
    # Feature 48-51: Horizontal discoid band energy (characteristic of linear plate atelectasis)
    for tier in range(4):
        # horizontal gradient across columns
        h_grad = abs(block_means[tier * 4 + 0] - block_means[tier * 4 + 1]) + \
                 abs(block_means[tier * 4 + 1] - block_means[tier * 4 + 2]) + \
                 abs(block_means[tier * 4 + 2] - block_means[tier * 4 + 3])
        features.append((h_grad - 20.0) / 20.0)

    # Feature 52-55: Vertical gradients across tiers
    for col in range(4):
        v_grad = abs(block_means[0 * 4 + col] - block_means[1 * 4 + col]) + \
                 abs(block_means[1 * 4 + col] - block_means[2 * 4 + col]) + \
                 abs(block_means[2 * 4 + col] - block_means[3 * 4 + col])
        features.append((v_grad - 20.0) / 20.0)

    # Feature 56-59: Discoid Atelectasis Ratio: ratio of horizontal sharpness to costophrenic fluid
    for tier in range(4):
        h_sharpness = abs(block_means[tier * 4 + 1] - block_means[tier * 4 + 2])
        cp_effusion_comp = (block_means[3 * 4 + 0] + block_means[3 * 4 + 3]) / 2.0 + 1e-5
        features.append((h_sharpness / cp_effusion_comp - 0.2) * 3.0)

    # Feature 60-63: Ipsilateral diaphragm elevation vs fluid meniscus
    features.append((block_means[2 * 4 + 0] - block_means[3 * 4 + 0]) / 20.0)
    features.append((block_means[2 * 4 + 3] - block_means[3 * 4 + 3]) / 20.0)
    features.append((block_stds[2 * 4 + 1] - block_stds[3 * 4 + 1]) / 15.0)
    features.append((block_stds[2 * 4 + 2] - block_stds[3 * 4 + 2]) / 15.0)

    return features[:64]

def softmax(logits, temp=1.2):
    scaled = [l / temp for l in logits]
    max_l = max(scaled)
    exps = [math.exp(x - max_l) for x in scaled]
    s = sum(exps)
    return [x / s for x in exps]

def load_split_data(split_name):
    data = []
    base = os.path.join("data", split_name)
    for c_idx, c_name in enumerate(CLASSES):
        sub = c_name.lower().replace(" ", "_")
        folder = os.path.join(base, sub)
        if not os.path.exists(folder):
            continue
        for f in glob.glob(os.path.join(folder, "*.png")):
            w, h, p = decode_png(f)
            feats = extract_radiograph_features(p, w, h)
            data.append((feats, c_idx, f))
    return data

print("Loading dataset splits...")
hosp_a_data = load_split_data("hospital_a")
hosp_b_data = load_split_data("hospital_b")
hosp_c_data = load_split_data("hospital_c")
val_data    = load_split_data("validation")
test_data   = load_split_data("test")

print(f"Loaded: HospA={len(hosp_a_data)}, HospB={len(hosp_b_data)}, HospC={len(hosp_c_data)}, Val={len(val_data)}, Test={len(test_data)}")

# Global weights initialization
num_classes = len(CLASSES)
num_features = 64
global_weights = [[0.0 for _ in range(num_features)] for _ in range(num_classes)]
global_bias = [0.0 for _ in range(num_classes)]

# Train client locally
def train_client(data, initial_weights, initial_bias, epochs=8, lr=0.03, weight_decay=0.001):
    w = [list(row) for row in initial_weights]
    b = list(initial_bias)
    
    for epoch in range(epochs):
        shuffled = list(data)
        random.shuffle(shuffled)
        for feats, label, _ in shuffled:
            # forward
            logits = [sum(feats[i] * w[c][i] for i in range(num_features)) + b[c] for c in range(num_classes)]
            probs = softmax(logits, temp=1.0)
            
            # gradient step with L2 regularization
            for c in range(num_classes):
                target = 1.0 if c == label else 0.0
                grad = probs[c] - target
                for i in range(num_features):
                    w[c][i] -= lr * (grad * feats[i] + weight_decay * w[c][i])
                b[c] -= lr * grad
    return w, b

def evaluate(data, w, b, temp=1.2):
    correct = 0
    total = len(data)
    per_class = {c: {"tp": 0, "total": 0} for c in CLASSES}
    
    for feats, label, _ in data:
        logits = [sum(feats[i] * w[c][i] for i in range(num_features)) + b[c] for c in range(num_classes)]
        probs = softmax(logits, temp=temp)
        pred = probs.index(max(probs))
        c_name = CLASSES[label]
        per_class[c_name]["total"] += 1
        if pred == label:
            correct += 1
            per_class[c_name]["tp"] += 1
            
    acc = correct / total * 100.0
    return acc, per_class

print("\n--- Commencing Federated Learning Rounds ---")
rounds = 6
for r in range(1, rounds + 1):
    # Hospital A local training
    w_a, b_a = train_client(hosp_a_data, global_weights, global_bias, epochs=8, lr=0.025)
    # Hospital B local training
    w_b, b_b = train_client(hosp_b_data, global_weights, global_bias, epochs=8, lr=0.025)
    # Hospital C local training
    w_c, b_c = train_client(hosp_c_data, global_weights, global_bias, epochs=8, lr=0.025)
    
    # FedAvg parameter aggregation
    n_a, n_b, n_c = len(hosp_a_data), len(hosp_b_data), len(hosp_c_data)
    total_n = n_a + n_b + n_c
    
    for c in range(num_classes):
        for i in range(num_features):
            global_weights[c][i] = (n_a * w_a[c][i] + n_b * w_b[c][i] + n_c * w_c[c][i]) / total_n
        global_bias[c] = (n_a * b_a[c] + n_b * b_b[c] + n_c * b_c[c]) / total_n
        
    val_acc, _ = evaluate(val_data, global_weights, global_bias)
    test_acc, per_class = evaluate(test_data, global_weights, global_bias)
    print(f"Round {r}/{rounds} -> Val Accuracy: {val_acc:.1f}%, Test Accuracy: {test_acc:.1f}%")

print("\nFinal Test Per-Class Performance:")
for c, st in per_class.items():
    print(f"  {c:<16}: {st['tp']}/{st['total']} ({st['tp']/st['total']*100:.1f}%)")

print("\nEvaluating the 6 Demonstration Benchmark Cases:")
sample_paths = [
    ("Normal", "data/test/normal/CXR_00541.png"),
    ("Pneumonia", "data/test/pneumonia/CXR_00561.png"),
    ("Atelectasis", "data/test/atelectasis/CXR_00581.png"),
    ("Cardiomegaly", "data/test/cardiomegaly/CXR_00601.png"),
    ("Pleural Effusion", "data/test/pleural_effusion/CXR_00621.png"),
    ("Infiltration", "data/test/infiltration/CXR_00641.png"),
]

for exp, p in sample_paths:
    w, h, px = decode_png(p)
    feats = extract_radiograph_features(px, w, h)
    logits = [sum(feats[i] * global_weights[c][i] for i in range(num_features)) + global_bias[c] for c in range(num_classes)]
    probs = softmax(logits, temp=1.2)
    pred_idx = probs.index(max(probs))
    pred_name = CLASSES[pred_idx]
    conf = probs[pred_idx] * 100.0
    status = "✓ MATCH" if pred_name == exp else "✗ MISMATCH"
    print(f"  Expected: {exp:<16} -> Predicted: {pred_name:<16} (Conf: {conf:5.1f}%) [{status}]")
