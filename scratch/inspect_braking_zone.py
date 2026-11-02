import cv2, os, sys, pickle
import numpy as np

with open('scratch/sinchana_bbreak_analysis.pkl', 'rb') as f:
    data = pickle.load(f)

print("Frames 270 to 350 granular inspection:")
print("Frame | Moto box (x1, y1, x2, y2) | cx, cy | Scale (h) | Disp (px) | dt (s) | Raw Speed (h/s) | Decel (h/s^2)")

prev_m = None
prev_t = None
prev_spd = None

for d in data:
    f = d['f']
    if not (270 <= f <= 350):
        continue
    m = d['moto']
    if not m:
        continue
    
    t = f / 30.0
    cx = (m[0] + m[2]) / 2.0
    cy = (m[1] + m[3]) / 2.0
    h = m[3] - m[1]
    
    if prev_m is not None:
        dt = t - prev_t
        disp = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        avg_h = (h + prev_h) / 2.0
        spd = (disp / avg_h) / dt
        decel = (prev_spd - spd) / dt if prev_spd is not None else 0.0
        print(f"F {f:03d} | ({m[0]:3d}, {m[1]:3d}, {m[2]:3d}, {m[3]:3d}) | ({cx:5.1f}, {cy:5.1f}) | h={h:3d} | disp={disp:4.1f}px | dt={dt:.3f}s | spd={spd:4.2f} h/s | decel={decel:5.2f}")
        prev_spd = spd
    else:
        prev_spd = 0.0
    
    prev_m = m
    prev_t = t
    prev_cx = cx
    prev_cy = cy
    prev_h = h
