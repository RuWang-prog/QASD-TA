import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 参数设置
C = 32

def sigmoid(x, k):
    return 1 / (1 + np.exp(-x / k))

def iou_same_square(s, d):
    """相同正方形框，水平偏移d，边长s，计算IoU"""
    if d >= s:
        return 0.0
    return (1 - d / s) ** 2

def nwd_same_square(d, C):
    """NWD只依赖偏移距离d和C，与边长无关"""
    return np.exp(-d / C)

def safit_same_square(s, d, C):
    """SAFit，权重基于第二个框面积（即s）"""
    w = sigmoid(s - C, C)          # 注意：原始公式中 sigmoid(sqrt(area)-C) = sigmoid(s - C)
    i = iou_same_square(s, d)
    n = nwd_same_square(d, C)
    return w * i + (1 - w) * n

# 边长范围
s_vals = np.linspace(1, 200, 500)   # 边长从1到200像素
d_fixed = [5, 10, 15, 20]           # 固定偏移量（像素）

plt.figure(figsize=(12, 8))

# 绘制IoU曲线（与边长无关？实际上IoU与s和d都有关）
for d in d_fixed:
    iou_vals = [iou_same_square(s, d) for s in s_vals]
    plt.plot(s_vals, iou_vals, '--', label=f'IoU (d={d}px)', linewidth=2)

# 绘制SAFit曲线
for d in d_fixed:
    safit_vals = [safit_same_square(s, d, C) for s in s_vals]
    plt.plot(s_vals, safit_vals, '-', label=f'SAFit (d={d}px)', linewidth=2)

# 标注面积区间
edge_limits = [(0, 8), (8, 16), (16, 32), (32, 96), (96, 200)]
labels = ['ex_tiny\n(0-8²)', 'tiny\n(8²-16²)', 'small\n(16²-32²)', 'medium\n(32²-96²)', 'large\n(96²-200²)']
colors = ['red', 'orange', 'green', 'blue', 'purple']

for (low, high), label, color in zip(edge_limits, labels, colors):
    plt.axvspan(low, high, alpha=0.15, color=color)
    mid = (low + high) / 2
    if mid < 200:
        plt.text(mid, 0.95, label, ha='center', fontsize=9, color=color)

plt.xlabel('框边长 (pixels)', fontsize=12)
plt.ylabel('相似度 (IoU / SAFit)', fontsize=12)
plt.title('固定水平偏移量下，IoU与SAFit随框边长的变化\n(SAFit使用第二个框面积权重，C=32)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.show()