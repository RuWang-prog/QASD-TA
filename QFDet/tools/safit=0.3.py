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
    w = sigmoid(s - C, C)          # 权重基于第二个框边长 s
    i = iou(s, d)
    n = nwd(d)
    return w * i + (1 - w) * n

def find_offset(s, target=0.3):
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
# 转换为 NumPy 数组，以便后续布尔索引
offsets = np.array([find_offset(s) for s in s_vals])

plt.figure(figsize=(10, 6))
plt.plot(s_vals, offsets, label='SAFit=0.3 (C=32)', color='blue')
plt.xlabel('框边长 (pixels)', fontsize=12)
plt.ylabel('水平偏移像素 (d)', fontsize=12)
plt.title('相同大小正方形框，SAFit=0.3时偏移与边长关系\n(权重基于第二个框面积，C=32)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 标注面积区间
edge_limits = [(0, 8), (8, 16), (16, 32), (32, 96), (96, 316)]
labels = ['ex_tiny (0-8²)', 'tiny (8²-16²)', 'small (16²-32²)', 'medium (32²-96²)', 'large (96²-1e5²)']
colors = ['red', 'orange', 'green', 'blue', 'purple']
# 计算有效偏移的最大值（忽略 nan）
valid_max = np.nanmax(offsets)  # 安全获取最大值
for (low, high), label, color in zip(edge_limits, labels, colors):
    plt.axvspan(low, high, alpha=0.2, color=color)
    mid = (low + high) / 2
    if mid < 320:
        plt.text(mid, valid_max * 0.9, label, ha='center', fontsize=8, color=color)

plt.legend()
plt.tight_layout()
plt.show()