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

def extract_features(pixels, width=224, height=224):
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
    right_cp = (block_means[3 * 4 + 0] - block_means[1 * 4 + 0]) / 25.0
    left_cp  = (block_means[3 * 4 + 3] - block_means[1 * 4 + 3]) / 25.0
    features.append(left_cp)
    features.append(right_cp)
    features.append(max(left_cp, right_cp))

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
    # Horizontal gradients (horizontal plate-like collapse / fluid levels)
    for tier in range(4):
        # horizontal gradient across columns
        h_grad = abs(block_means[tier * 4 + 0] - block_means[tier * 4 + 1]) + \
                 abs(block_means[tier * 4 + 1] - block_means[tier * 4 + 2]) + \
                 abs(block_means[tier * 4 + 2] - block_means[tier * 4 + 3])
        features.append((h_grad - 20.0) / 20.0)

    # Vertical gradients across tiers
    for col in range(4):
        v_grad = abs(block_means[0 * 4 + col] - block_means[1 * 4 + col]) + \
                 abs(block_means[1 * 4 + col] - block_means[2 * 4 + col]) + \
                 abs(block_means[2 * 4 + col] - block_means[3 * 4 + col])
        features.append((v_grad - 20.0) / 20.0)

    # Texture ratios (perihilar vs periphery)
    for tier in range(4):
        center_std = (block_stds[tier * 4 + 1] + block_stds[tier * 4 + 2]) / 2.0
        edge_std = (block_stds[tier * 4 + 0] + block_stds[tier * 4 + 3]) / 2.0 + 1e-5
        features.append((center_std / edge_std - 1.0))

    # Density contrast ratios
    for col in range(4):
        top_m = block_means[0 * 4 + col] + 1e-5
        bot_m = block_means[3 * 4 + col]
        features.append((bot_m / top_m - 1.0))

    return features[:64]

print("Feature extractor defined. Total features:", len(extract_features([100] * (224*224))))
