import cv2, os, sys, pickle
import numpy as np

with open('scratch/sinchana_bbreak_analysis.pkl', 'rb') as f:
    data = pickle.load(f)

# Let's inspect all frames where speed > 15 px/s and foot distance is low
print("Frames where Moto is MOVING and foot is close to ground:")
print("Frame | Speed (px/s) | Moto Height | L_dist | R_dist | Asymmetry | Lowest Foot")

for i in range(1, len(data)):
    d = data[i]
    prev_d = data[i-1]
    f = d['f']
    m = d['moto']
    pm = prev_d['moto']
    if not (m and pm):
        continue
    
    cx = (m[0] + m[2]) / 2.0
    cy = (m[1] + m[3]) / 2.0
    pcx = (pm[0] + pm[2]) / 2.0
    pcy = (pm[1] + pm[3]) / 2.0
    dt = (d['f'] - prev_d['f']) / 30.0
    disp = np.sqrt((cx - pcx)**2 + (cy - pcy)**2) / dt
    
    moto_b = m[3]
    moto_h = m[3] - m[1]
    
    l_pts = [p for p in [d['l_ankle'], d['l_heel'], d['l_toe']] if p and p[2] > 0.3]
    r_pts = [p for p in [d['r_ankle'], d['r_heel'], d['r_toe']] if p and p[2] > 0.3]
    l_low = max([p[1] for p in l_pts]) if l_pts else 0
    r_low = max([p[1] for p in r_pts]) if r_pts else 0
    
    ld = moto_b - l_low if l_low > 0 else 999
    rd = moto_b - r_low if r_low > 0 else 999
    asym = abs(l_low - r_low)
    
    # Check if foot is within 22% of moto height or <= 30px
    foot_threshold = max(18.0, moto_h * 0.15)
    
    if (ld <= foot_threshold or rd <= foot_threshold) and disp > 15.0:
        which = 'Right' if rd < ld else 'Left'
        min_d = min(ld, rd)
        print(f"F {f:03d} | spd: {disp:5.1f} px/s | moto_h: {moto_h:3d} | L:{ld:4.1f} R:{rd:4.1f} | asym: {asym:4.1f} | {which} (dist={min_d:4.1f}, thresh={foot_threshold:4.1f})")
