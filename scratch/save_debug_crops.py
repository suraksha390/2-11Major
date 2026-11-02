import cv2, os

video_path = r'static/uploads/videos/test_CAND-2026-083_sinchana_bbreak.mp4'
cap = cv2.VideoCapture(video_path)

out_dir = r'scratch/foot_debug_crops'
os.makedirs(out_dir, exist_ok=True)

f_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    f_idx += 1
    if (105 <= f_idx <= 125 and f_idx % 3 == 0) or (180 <= f_idx <= 200 and f_idx % 3 == 0):
        cv2.imwrite(os.path.join(out_dir, f"frame_{f_idx:03d}.jpg"), frame)

cap.release()
print("Saved debug crops to", out_dir)
