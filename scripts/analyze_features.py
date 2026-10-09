import sys
sys.path.insert(0, ".")
import os, glob
from backend.xray.dataset import decode_png

classes = ["Normal", "Pneumonia", "Atelectasis", "Cardiomegaly", "Pleural Effusion", "Infiltration"]

def get_stats(pixels, width=224, height=224):
    grid = [pixels[r*width:(r+1)*width] for r in range(height)]
    mean_val = sum(pixels) / len(pixels)
    upper = [grid[r][c] for r in range(20, 80) for c in range(25, 195)]
    lower = [grid[r][c] for r in range(140, 205) for c in range(25, 195)]
    left_lung = [grid[r][c] for r in range(60, 170) for c in range(25, 95)]
    right_lung = [grid[r][c] for r in range(60, 170) for c in range(125, 195)]
    left_mean = sum(left_lung) / len(left_lung)
    right_mean = sum(right_lung) / len(right_lung)
    cardiac_zone = [grid[r][c] for r in range(100, 175) for c in range(75, 150)]
    left_cp = [grid[r][c] for r in range(175, 215) for c in range(15, 60)]
    right_cp = [grid[r][c] for r in range(175, 215) for c in range(160, 205)]
    
    return {
        "mean": mean_val,
        "upper": sum(upper) / len(upper),
        "lower": sum(lower) / len(lower),
        "asym": abs(left_mean - right_mean),
        "cardiac": sum(cardiac_zone) / len(cardiac_zone),
        "cp": (sum(left_cp) + sum(right_cp)) / (len(left_cp) + len(right_cp))
    }

for c in classes:
    sub = c.lower().replace(" ", "_")
    files = glob.glob("data/test/" + sub + "/*.png")
    all_stats = [get_stats(decode_png(f)[2]) for f in files]
    avg = {k: sum(s[k] for s in all_stats)/len(all_stats) for k in all_stats[0]}
    print("%-16s: Mean=%5.1f, Upper=%5.1f, Lower=%5.1f, Asym=%4.1f, Cardiac=%5.1f, CP=%5.1f" % (
        c, avg["mean"], avg["upper"], avg["lower"], avg["asym"], avg["cardiac"], avg["cp"]
    ))
