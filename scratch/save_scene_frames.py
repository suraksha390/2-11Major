import cv2
import numpy as np

cap = cv2.VideoCapture("without_helmet_foottouch.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Let's save a mosaic or contact sheet of key frames to see the whole video
frames_to_save = [30, 60, 90, 120, 150, 165, 175, 185, 200, 220, 250, 280]

idx = 0
saved = {}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1
    if idx in frames_to_save:
        saved[idx] = frame

cap.release()

for f_idx, img in saved.items():
    cv2.imwrite(f"scratch/scene_frame_{f_idx:03d}.jpg", img)

print(f"Saved {len(saved)} scene frames to scratch/")
