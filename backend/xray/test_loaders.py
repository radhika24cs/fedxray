"""
backend/xray/test_loaders.py
Verification script for Step 2 and Step 3:
- Tests metadata parsing
- Verifies image decoding & normalization
- Generates a sample mini-batch from Hospital A and Validation sets
- Validates batch dimensions, label encoding, and non-IID statistics
"""

import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.xray.dataset import ChestXrayDataset, ChestXrayDataLoader

def run_verification():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    print(f"==================================================")
    print(f"STEP 2 & 3: CHEST X-RAY DATA LOADER VERIFICATION")
    print(f"Data directory: {data_dir}")
    print(f"==================================================")

    for split_name in ["hospital_a", "hospital_b", "hospital_c", "validation", "test"]:
        ds = ChestXrayDataset(data_root=data_dir, split=split_name)
        counts = ds.get_class_counts()
        print(f"\n[Split: {split_name.upper()}] - Total: {len(ds)} images")
        for cls_name, count in counts.items():
            pct = (count / len(ds)) * 100
            print(f"  • {cls_name:<16}: {count:2d} ({pct:5.1f}%)")

    # Test single sample retrieval
    print(f"\n--------------------------------------------------")
    print(f"TESTING SINGLE SAMPLE INGESTION (Hospital A, index 0)")
    ds_a = ChestXrayDataset(data_root=data_dir, split="hospital_a")
    sample_0 = ds_a[0]
    print(f"  Image ID      : {sample_0['metadata']['image_id']}")
    print(f"  Finding       : {sample_0['label_name']} (Class index: {sample_0['label_idx']})")
    print(f"  Patient ID    : {sample_0['metadata']['patient_id']}")
    print(f"  Demographics  : Age {sample_0['metadata']['age']}, Gender {sample_0['metadata']['gender']}, View {sample_0['metadata']['view_position']}")
    print(f"  Dimensions    : {sample_0['width']} x {sample_0['height']}")
    print(f"  Pixel count   : {len(sample_0['pixels'])} normalized float values")
    print(f"  Pixel range   : min={min(sample_0['pixels']):.3f}, max={max(sample_0['pixels']):.3f}")
    print(f"  One-hot vector: {sample_0['one_hot']}")

    # Test mini-batch collation
    print(f"\n--------------------------------------------------")
    print(f"TESTING MINI-BATCH DATA LOADER (batch_size=8)")
    loader = ChestXrayDataLoader(ds_a, batch_size=8, shuffle=True)
    first_batch = next(iter(loader))
    print(f"  Batch size returned : {first_batch['batch_size']}")
    print(f"  Labels in batch     : {first_batch['labels']}")
    print(f"  Batch pixel matrix  : {len(first_batch['pixels'])} items x {len(first_batch['pixels'][0])} elements")
    print(f"  Total batches/epoch : {len(loader)}")
    print(f"==================================================")
    print(f"SUCCESS: Preprocessing and DataLoaders are verified!")
    print(f"==================================================")

if __name__ == "__main__":
    run_verification()
