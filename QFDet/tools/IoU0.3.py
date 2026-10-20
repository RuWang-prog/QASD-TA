import numpy as np
import matplotlib.pyplot as plt

# IoU=0.3 对应的偏移比例
k = 1 - np.sqrt(0.3)  # ≈ 0.452277

# 定义面积范围对应的边长范围（sqrt(area)）
# 面积: [0, 8^2], [8^2, 16^2], [16^2, 32^2], [32^2, 96^2], [96^2, 1e5^2]
# 边长: [0,8], [8,16], [16,32], [32,96], [96, sqrt(1e5)=316.23]
edge_limits = [(0, 8), (8, 16), (16, 32), (32, 96), (96, 316.23)]
labels = ['ex_tiny (0-8²)', 'tiny (8²-16²)', 'small (16²-32²)', 'medium (32²-96²)', 'large (96²-1e5²)']

# 生成每个范围内的边长采样点
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange', 'red', 'purple']

for (low, high), label, color in zip(edge_limits, labels, colors):
    edges = np.linspace(low, high, 100)
    offsets = k * edges
    plt.plot(edges, offsets, label=label, color=color, linewidth=2)

plt.xlabel('框边长 (pixels)', fontsize=12)
plt.ylabel('水平偏移像素 (d)', fontsize=12)
plt.title('IoU=0.3 时相同大小正方形框的偏移与边长关系', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()