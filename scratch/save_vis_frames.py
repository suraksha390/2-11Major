import cv2, os

video_path = r'static/uploads/videos/test_CAND-2026-083_sinchana_bbreak.mp4'
cap = cv2.VideoCapture(video_path)

out_dir = r'scratch/vis_inspect'
os.makedirs(out_dir, exist_ok=True)

# Frames to save:
# 10, 115, 185, 195, 260, 310, 380, 400, 420
frames_to_save = [10, 60, 100, 115, 125, 175, 185, 195, 205, 255, 265, 300, 360, 380, 395, 405, 420, 440]

f_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    f_idx += 1
    if f_idx in frames_to_save:
        cv2.imwrite(os.path.join(out_dir, f"frame_{f_idx:03d}.jpg"), frame)

cap.release()
print("Saved inspection frames to", out_dir)
