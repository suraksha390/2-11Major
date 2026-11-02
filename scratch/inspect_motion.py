import cv2
import pickle

with open('scratch/sinchana_bbreak_analysis.pkl', 'rb') as f:
    data = pickle.load(f)

# Let's check motorcycle x, y position over time
print("Frame | Moto Box (x1, y1, x2, y2) | Moto Center X, Y | Width, Height")
for d in data[::8]:
    f = d['f']
    m = d['moto']
    if m:
        cx = (m[0] + m[2]) / 2.0
        cy = (m[1] + m[3]) / 2.0
        w = m[2] - m[0]
        h = m[3] - m[1]
        print(f"F {f:03d} | ({m[0]:3d}, {m[1]:3d}, {m[2]:3d}, {m[3]:3d}) | cx={cx:5.1f}, cy={cy:5.1f} | {w:3d}x{h:3d}")
    else:
        print(f"F {f:03d} | None")
