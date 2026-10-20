import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

C = 32

def sigmoid(x, k):
    return 1 / (1 + np.exp(-x / k))

def iou(s, d):
    if d > s:
        return 0.0
    return (1 - d / s) ** 2

def nwd(d):
    return np.exp(-d / C)

def safit(s, d):
    w = sigmoid(s - C, C)
    i = iou(s, d)
    n = nwd(d)
    return w * i + (1 - w) * n

def find_offset_safit(s, target=0.3):
    if s <= 0:
        return np.nan
    if safit(s, 0) < target:
        return 0.0
    if safit(s, s) > target:
        return s
    def func(d):
        return safit(s, d) - target
    d_sol = fsolve(func, x0=s/2, full_output=False)[0]
    return np.clip(d_sol, 0, s)

# 边长范围
s_vals = np.linspace(0.1, 320, 500)

# IoU=0.3 偏移（线性）
k_iou = 1 - np.sqrt(0.3)
offsets_iou = k_iou * s_vals

# SAFit=0.3 偏移（数值求解）
offsets_safit = np.array([find_offset_safit(s) for s in s_vals])

plt.figure(figsize=(10, 6))
plt.plot(s_vals, offsets_iou, label='IoU=0.3', color='blue', linestyle='-', linewidth=2)
plt.plot(s_vals, offsets_safit, label='SAFit=0.3 (C=32)', color='red', linestyle='--', linewidth=2)

plt.xlabel('框边长 (pixels)', fontsize=12)
plt.ylabel('水平偏移像素 (d)', fontsize=12)
plt.title('相同大小正方形框，IoU=0.3 与 SAFit=0.3 偏移对比', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 标注面积区间
edge_limits = [(0, 8), (8, 16), (16, 32), (32, 96), (96, 316)]
labels = ['ex_tiny (0-8²)', 'tiny (8²-16²)', 'small (16²-32²)', 'medium (32²-96²)', 'large (96²-1e5²)']
colors = ['red', 'orange', 'green', 'blue', 'purple']
valid_max = np.nanmax(offsets_safit)  # 用于文字垂直位置
for (low, high), label, color in zip(edge_limits, labels, colors):
    plt.axvspan(low, high, alpha=0.2, color=color)
    mid = (low + high) / 2
    if mid < 320:
        plt.text(mid, valid_max * 0.9, label, ha='center', fontsize=8, color=color)

plt.legend()
plt.tight_layout()
plt.show()