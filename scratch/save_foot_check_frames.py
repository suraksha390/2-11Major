import cv2
import mediapipe as mp
import numpy as np

# Let's crop the rider and lower body at frames 96, 102, 108, 160, 170, 180
cap = cv2.VideoCapture("without_helmet_foottouch.mp4")
frames_to_save = [90, 96, 102, 108, 114, 150, 165, 170, 175, 180, 190]

idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1
    if idx in frames_to_save:
        cv2.imwrite(f"scratch/foot_check_f{idx:03d}.jpg", frame)
cap.release()
print("Saved foot check frames")
