import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

# Test a few crops that were wrongly marked as helmet PASS
crop_names = [
    "scratch/head_crops_test/f060_helmet_PASS_0.90.jpg",
    "scratch/head_crops_test/f090_helmet_PASS_0.90.jpg",
    "scratch/head_crops_test/f120_helmet_PASS_0.90.jpg",
    "scratch/head_crops_test/f170_no_helmet_FAIL_0.31.jpg",
    "scratch/head_crops_test/f195_helmet_PASS_0.90.jpg"
]

for cp in crop_names:
    if not os.path.exists(cp):
        continue
    head_roi = cv2.imread(cp)
    h, w = head_roi.shape[:2]
    gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(head_roi, cv2.COLOR_BGR2YCrCb)
    cr = ycbcr[:, :, 1]
    cb = ycbcr[:, :, 2]
    b, g, r = head_roi[:, :, 0], head_roi[:, :, 1], head_roi[:, :, 2]
    total_pixels = float(h * w + 1e-5)

    skin_mask = ((cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 125) &
                 (r > g + 8) & (g > b + 2))
    skin_ratio = float(np.sum(skin_mask)) / total_pixels

    mid_y = int(h * 0.45)
    crown = head_roi[:mid_y, :]
    face = head_roi[mid_y:, :]
    crown_skin = float(np.sum(skin_mask[:mid_y, :])) / float(crown.size // 3 + 1e-5)
    face_skin = float(np.sum(skin_mask[mid_y:, :])) / float(face.size // 3 + 1e-5)

    crown_hsv = cv2.cvtColor(crown, cv2.COLOR_BGR2HSV)
    highlight_mask = (crown_hsv[:, :, 2] >= 180) & (crown_hsv[:, :, 1] <= 45)
    specular_ratio = float(np.mean(highlight_mask))

    hair_mask = (crown_hsv[:, :, 2] < 55) & (crown_hsv[:, :, 1] < 70)
    hair_ratio = float(np.mean(hair_mask))

    print(f"\n--- {cp} ({w}x{h}, lap={lap_var:.1f}) ---")
    print(f"skin_ratio: {skin_ratio:.3f}, crown_skin: {crown_skin:.3f}, face_skin: {face_skin:.3f}")
    print(f"specular_ratio: {specular_ratio:.3f}, hair_ratio: {hair_ratio:.3f}")
    
    # Check why is_helmet was True:
    is_bare_head = (
        (crown_skin > 0.15 and face_skin > 0.20) or
        (hair_ratio > 0.30 and face_skin > 0.22) or
        (skin_ratio > 0.28 and crown_skin > 0.10)
    )
    print(f"is_bare_head: {is_bare_head}")
    print(f"Conditions: crown_skin>0.15 and face_skin>0.20: {crown_skin > 0.15 and face_skin > 0.20}")
    print(f"Conditions: hair_ratio>0.30 and face_skin>0.22: {hair_ratio > 0.30 and face_skin > 0.22}")
    print(f"Conditions: skin_ratio>0.28 and crown_skin>0.10: {skin_ratio > 0.28 and crown_skin > 0.10}")
