import cv2
import os

cap = cv2.VideoCapture("without_helmet_foottouch.mp4")
out_dir = "scratch/video_frames"
os.makedirs(out_dir, exist_ok=True)

# Save every 10th frame
idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if idx % 10 == 0:
        cv2.imwrite(f"{out_dir}/frame_{idx:03d}.jpg", frame)
    idx += 1

cap.release()
print(f"Extracted {idx} frames to {out_dir}")
