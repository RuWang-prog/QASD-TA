import numpy as np
import matplotlib.pyplot as plt


def compute_iou(box, boxes):
    """计算单个框与一组框的 IoU (格式: x1, y1, x2, y2)"""
    if len(boxes) == 0:
        return np.array([])

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    return inter_area / (union_area + 1e-7)


def evaluate_f1_curve(dataset_results, iou_thresh=0.5):
    """
    高效计算并绘制 Precision, Recall, F1 随 置信度阈值 变化的曲线。

    参数:
        dataset_results: 列表，每个元素代表一张图的结果字典
                         格式: {'gt_boxes': ndarray, 'pred_boxes': ndarray, 'pred_scores': ndarray}
        iou_thresh: 判断 TP 的 IoU 阈值 (通常为 0.5)
    """
    all_scores = []
    all_matches = []  # 1 代表 TP, 0 代表 FP
    total_gts = 0

    print("🚀 正在进行贪心匹配 (Greedy Matching)...")
    for res in dataset_results:
        gts = res['gt_boxes']
        preds = res['pred_boxes']
        scores = res['pred_scores']

        total_gts += len(gts)

        if len(preds) == 0:
            continue

        # 按得分降序排列预测框
        sort_idx = np.argsort(scores)[::-1]
        preds = preds[sort_idx]
        scores = scores[sort_idx]

        matched_gt_indices = set()

        for i, pred_box in enumerate(preds):
            all_scores.append(scores[i])

            if len(gts) == 0 or len(matched_gt_indices) == len(gts):
                all_matches.append(0)  # 所有的 GT 都被匹配完了，剩下的全是 FP
                continue

            ious = compute_iou(pred_box, gts)
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]

            # 如果 IoU 达标，且该真实框尚未被认领，则为 TP
            if best_iou >= iou_thresh and best_iou_idx not in matched_gt_indices:
                all_matches.append(1)
                matched_gt_indices.add(best_iou_idx)
            else:
                all_matches.append(0)  # FP

    if total_gts == 0 or len(all_scores) == 0:
        print("❌ 数据集为空或没有预测结果！")
        return

    print("📈 正在计算指标曲线...")
    # 转换为 Numpy 数组并按照分数全局降序排列
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)

    global_sort_idx = np.argsort(all_scores)[::-1]
    all_scores = all_scores[global_sort_idx]
    all_matches = all_matches[global_sort_idx]

    # 使用累加和 (Cumsum) 极速计算各个截断点下的 TP 和 FP 总数
    acc_tp = np.cumsum(all_matches)
    acc_fp = np.cumsum(1 - all_matches)

    # 计算 Precision 和 Recall
    precision = acc_tp / (acc_tp + acc_fp + 1e-7)
    recall = acc_tp / (total_gts + 1e-7)

    # 计算 F1-Score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)

    # 找到 F1-Score 最高的点 (最佳操作点)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = all_scores[best_idx]
    best_p = precision[best_idx]
    best_r = recall[best_idx]

    print("-" * 30)
    print(f"🏆 最佳部署指标 (IoU > {iou_thresh}):")
    print(f"推荐部署阈值 (Score Thresh): {best_thresh:.3f}")
    print(f"最高 F1-Score: {best_f1:.3f}")
    print(f"此时 Precision: {best_p:.3f} | Recall: {best_r:.3f}")
    print("-" * 30)

    # ================= 绘图 =================
    plt.figure(figsize=(10, 6))

    # 画曲线
    plt.plot(all_scores, f1_scores, label='F1-Score', color='green', linewidth=2)
    plt.plot(all_scores, precision, label='Precision', color='blue', linestyle='--')
    plt.plot(all_scores, recall, label='Recall', color='red', linestyle='--')

    # 标出最优点
    plt.scatter(best_thresh, best_f1, color='gold', s=100, zorder=5, edgecolor='black',
                label=f'Best F1 ({best_f1:.2f} @ {best_thresh:.2f})')
    plt.axvline(x=best_thresh, color='gray', linestyle=':', alpha=0.7)

    plt.title(f'Detection Metrics vs. Confidence Threshold (IoU={iou_thresh})', fontsize=14)
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    # 因为得分是降序的，让 X 轴从 0 到 1 正向显示更符合直觉
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()


# ================= 测试代码 =================
if __name__ == "__main__":
    # 生成一些模拟测试数据
    np.random.seed(42)
    dummy_dataset = []

    for _ in range(50):  # 模拟 50 张图片
        # 模拟 3 个真实的微小目标
        gts = np.array([
            [10, 10, 20, 20],
            [50, 50, 60, 60],
            [100, 100, 110, 110]
        ])

        # 模拟预测框 (包含定位准的高分框、定位准的低分框，以及定位差的冗余高分框)
        preds = gts + np.random.normal(0, 2, size=gts.shape)  # 加点像素偏移
        scores = np.random.uniform(0.7, 0.99, size=3)  # 真实目标的得分较高

        # 强行塞入一些冗余框 (FP)
        noise_boxes = np.array([[12, 12, 22, 22], [80, 80, 90, 90]])
        noise_scores = np.array([0.85, 0.4])  # 一个讨厌的高分冗余，一个低分冗余

        preds = np.vstack([preds, noise_boxes])
        scores = np.concatenate([scores, noise_scores])

        dummy_dataset.append({
            'gt_boxes': gts,
            'pred_boxes': preds,
            'pred_scores': scores
        })

    evaluate_f1_curve(dummy_dataset, iou_thresh=0.5)