"""
backend/xray/dataset.py
Dataset Loader and Batch Generator for Federated Chest X-ray Disease Detection.

Supports:
- Institutional partition loading (Hospital A, Hospital B, Hospital C)
- Central validation and global held-out test splits
- Pure Python PNG decompression (zero external dependency barrier)
- Metadata extraction (patient age, gender, view position PA/AP)
- Multi-class label encoding and batch generation
"""

import os
import csv
import json
import zlib
import struct
from typing import List, Dict, Any, Tuple, Optional
from backend.xray.preprocessing import normalize_intensity, TARGET_WIDTH, TARGET_HEIGHT

def decode_png(file_path: str) -> Tuple[int, int, List[int]]:
    """
    Decodes a PNG (grayscale, RGB, or RGBA) into width, height, and raw pixel list [0..255].
    Properly reverses all 5 standard PNG scanline filter types (None, Sub, Up, Average, Paeth).
    """
    with open(file_path, "rb") as f:
        data = f.read()

    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f"Invalid PNG header: {file_path}")

    pos = 8
    width = 0
    height = 0
    bit_depth = 8
    color_type = 0
    idat_chunks = []

    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos+4])[0]
        tag = data[pos+4:pos+8]
        chunk_data = data[pos+8:pos+8+length]
        pos += 12 + length

        if tag == b'IHDR':
            width, height, bit_depth, color_type, comp, flt, interl = struct.unpack(">IIBBBBB", chunk_data)
        elif tag == b'IDAT':
            idat_chunks.append(chunk_data)
        elif tag == b'IEND':
            break

    raw_decompressed = zlib.decompress(b"".join(idat_chunks))

    # Determine channels / bytes per pixel
    # color_type: 0=Grayscale, 2=RGB, 3=Indexed, 4=Grayscale+Alpha, 6=RGBA
    channels = 1
    if color_type == 2:
        channels = 3
    elif color_type == 4:
        channels = 2
    elif color_type == 6:
        channels = 4

    bytes_per_pixel = max(1, (channels * bit_depth) // 8)
    stride = width * bytes_per_pixel
    line_len = stride + 1

    prev_row = [0] * stride
    recon_bytes = []

    for y in range(height):
        filt = raw_decompressed[y * line_len]
        curr_row = [0] * stride
        for x in range(stride):
            raw = raw_decompressed[y * line_len + 1 + x]
            left = curr_row[x - bytes_per_pixel] if x >= bytes_per_pixel else 0
            up = prev_row[x]
            up_left = prev_row[x - bytes_per_pixel] if x >= bytes_per_pixel else 0

            if filt == 0:
                v = raw
            elif filt == 1:
                v = (raw + left) & 0xFF
            elif filt == 2:
                v = (raw + up) & 0xFF
            elif filt == 3:
                v = (raw + ((left + up) >> 1)) & 0xFF
            elif filt == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                pr = left if (pa <= pb and pa <= pc) else (up if pb <= pc else up_left)
                v = (raw + pr) & 0xFF
            else:
                v = raw

            curr_row[x] = v

        recon_bytes.extend(curr_row)
        prev_row = curr_row

    # Convert to grayscale [0..255] if RGB or RGBA
    if channels == 1:
        pixels = [int(b) for b in recon_bytes]
    elif channels == 3:
        pixels = []
        for i in range(0, len(recon_bytes), 3):
            r = recon_bytes[i]
            g = recon_bytes[i+1]
            b = recon_bytes[i+2]
            pixels.append(int(0.299 * r + 0.587 * g + 0.114 * b))
    elif channels == 4:
        pixels = []
        for i in range(0, len(recon_bytes), 4):
            r = recon_bytes[i]
            g = recon_bytes[i+1]
            b = recon_bytes[i+2]
            pixels.append(int(0.299 * r + 0.587 * g + 0.114 * b))
    else:
        pixels = [int(b) for b in recon_bytes[:width * height]]

    return width, height, pixels

class ChestXrayDataset:
    """
    Encapsulates a hospital or evaluation partition of Chest X-rays.
    """
    def __init__(self, data_root: str, split: str = "hospital_a", transform=True):
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.transform = transform
        
        self.classes = [
            "Normal",
            "Pneumonia",
            "Atelectasis",
            "Cardiomegaly",
            "Pleural Effusion",
            "Infiltration"
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Dict[str, Any]] = []
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self):
        metadata_csv = os.path.join(self.data_root, "metadata.csv")
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Missing metadata.csv at {metadata_csv}")

        with open(metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["hospital_split"] == self.split:
                    full_img_path = os.path.join(self.data_root, row["file_path"])
                    self.samples.append({
                        "image_id": row["image_id"],
                        "file_path": full_img_path,
                        "label_name": row["finding_label"],
                        "label_idx": self.class_to_idx[row["finding_label"]],
                        "category": row["category"],
                        "patient_id": row["patient_id"],
                        "age": int(row["patient_age"]),
                        "gender": row["patient_gender"],
                        "view_position": row["view_position"],
                        "institution": row["institution_name"]
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def get_class_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in self.classes}
        for s in self.samples:
            counts[s["label_name"]] += 1
        return counts

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self._cache:
            return self._cache[idx]

        sample = self.samples[idx]
        w, h, raw_pixels = decode_png(sample["file_path"])
        
        # Normalized float pixel vector
        norm_pixels = [normalize_intensity(p) for p in raw_pixels]
        
        # One-hot encoded ground truth
        one_hot = [0.0] * len(self.classes)
        one_hot[sample["label_idx"]] = 1.0

        item = {
            "pixels": norm_pixels,
            "width": w,
            "height": h,
            "label_idx": sample["label_idx"],
            "label_name": sample["label_name"],
            "one_hot": one_hot,
            "metadata": sample
        }
        self._cache[idx] = item
        return item

class ChestXrayDataLoader:
    """
    Mini-batch generator for local hospital training or evaluation.
    """
    def __init__(self, dataset: ChestXrayDataset, batch_size: int = 16, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            batch_idxs = self.indices[i:i + self.batch_size]
            batch_items = [self.dataset[idx] for idx in batch_idxs]
            
            # Collate batch
            batch_pixels = [item["pixels"] for item in batch_items]
            batch_labels = [item["label_idx"] for item in batch_items]
            batch_one_hot = [item["one_hot"] for item in batch_items]
            batch_meta = [item["metadata"] for item in batch_items]

            yield {
                "batch_size": len(batch_items),
                "pixels": batch_pixels,
                "labels": batch_labels,
                "one_hot": batch_one_hot,
                "metadata": batch_meta
            }

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
