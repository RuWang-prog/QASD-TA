# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from email.policy import default
import os
import os.path as osp
from signal import default_int_handler
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)
import numpy as np
import matplotlib.pyplot as plt
import os

#wr20260515 safit-based F1-score
def calc_safit_aligned(pred_boxes, gt_boxes, C_nwd=32.0):
    """
    100% 像素级对齐 AP 评估代码中的 Safit 计算逻辑。
    包含: IoU 计算, 欧氏距离 NWD, 基于 GT 面积的动态权重 w, 以及 IoU > 0 的物理安全阀。

    Args:
        pred_boxes: (N, 4) [x1, y1, x2, y2]
        gt_boxes: (M, 4) [x1, y1, x2, y2]
        C_nwd: 尺度常数 (默认 32.0)
    Returns:
        safit_matrix: (N, M) 融合相似度矩阵
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.array([])

    N, M = pred_boxes.shape[0], gt_boxes.shape[0]

    # --- 1. 计算原始 IoU 矩阵 ---
    x1 = np.maximum(pred_boxes[:, 0:1], gt_boxes[:, 0:1].T)
    y1 = np.maximum(pred_boxes[:, 1:2], gt_boxes[:, 1:2].T)
    x2 = np.minimum(pred_boxes[:, 2:3], gt_boxes[:, 2:3].T)
    y2 = np.minimum(pred_boxes[:, 3:4], gt_boxes[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # 转换为 [w, h]
    pred_w = np.maximum(pred_boxes[:, 2] - pred_boxes[:, 0], 1e-3)
    pred_h = np.maximum(pred_boxes[:, 3] - pred_boxes[:, 1], 1e-3)
    gt_w = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], 1e-3)
    gt_h = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1], 1e-3)

    area_pred = pred_w * pred_h
    area_gt = gt_w * gt_h

    union = area_pred[:, None] + area_gt[None, :] - inter
    iou_matrix = inter / (union + 1e-7)

    # --- 2. 计算 NWD 矩阵 (复刻欧氏距离版) ---
    cx_pred = pred_boxes[:, 0] + pred_w / 2
    cy_pred = pred_boxes[:, 1] + pred_h / 2
    half_w_pred = pred_w / 2
    half_h_pred = pred_h / 2
    vec_pred = np.stack([cx_pred, cy_pred, half_w_pred, half_h_pred], axis=1)  # (N, 4)

    cx_gt = gt_boxes[:, 0] + gt_w / 2
    cy_gt = gt_boxes[:, 1] + gt_h / 2
    half_w_gt = gt_w / 2
    half_h_gt = gt_h / 2
    vec_gt = np.stack([cx_gt, cy_gt, half_w_gt, half_h_gt], axis=1)  # (M, 4)

    vec_pred_exp = np.expand_dims(vec_pred, 1)  # (N, 1, 4)
    vec_gt_exp = np.expand_dims(vec_gt, 0)  # (1, M, 4)

    diff = vec_pred_exp - vec_gt_exp
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N, M)
    nwd_matrix = np.exp(-dist / C_nwd)

    # --- 3. 计算基于 GT 面积的动态权重 w ---
    Area_gt_matrix = np.tile(np.abs(area_gt), (N, 1))  # (N, M)
    # 对齐你的 sigmoid 函数: 1/(1+np.exp(-(1/C) * (sqrt(Area) - C)))
    x = np.sqrt(Area_gt_matrix) - C_nwd
    w = 1 / (1 + np.exp(-(1 / C_nwd) * x))

    # --- 4. 融合 Safit 相似度 ---
    safit_matrix = w * iou_matrix + (1 - w) * nwd_matrix

    # --- 5. 核心物理安全阀 (一票否决) ---
    safit_matrix[iou_matrix <= 1e-5] = 0.0

    return safit_matrix


def compute_and_plot_safit_f1(dataset, outputs, save_dir, match_thresh=0.5, C_nwd=32.0):
    """
    针对 MMDetection 格式，基于 Safit 相似度的 F1-Score 曲线评估器。
    """
    print("\n" + "=" * 50)
    print(f"🚀 启动工业级 F1-Score 分析 (基于 Safit > {match_thresh})...")
    print("=" * 50)

    all_scores = []
    all_matches = []  # 1 代表 TP, 0 代表 FP
    total_gts = 0

    # 1. 遍历数据集，提取 GT 和 Preds
    for i in range(len(dataset)):
        # 获取真实标签 GT
        ann_info = dataset.get_ann_info(i)
        gt_bboxes = ann_info['bboxes']
        total_gts += len(gt_bboxes)

        # 获取预测结果
        img_preds = outputs[i]
        pred_boxes = []
        pred_scores = []
        for cls_preds in img_preds:
            if len(cls_preds) > 0:
                pred_boxes.append(cls_preds[:, :4])
                pred_scores.append(cls_preds[:, 4])

        if len(pred_boxes) > 0:
            pred_boxes = np.vstack(pred_boxes)
            pred_scores = np.concatenate(pred_scores)

            # 按得分降序
            sort_idx = np.argsort(pred_scores)[::-1]
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]
        else:
            pred_boxes = np.empty((0, 4))
            pred_scores = np.empty((0,))

        # 2. 图片内贪心匹配 (使用对齐的 Safit)
        if len(pred_boxes) > 0 and len(gt_bboxes) > 0:
            # 极速计算：一次性算出整张图的 Safit 相似度矩阵 (N, M)
            safit_matrix = calc_safit_aligned(pred_boxes, gt_bboxes, C_nwd=C_nwd)

        matched_gts = set()
        for p_idx, score in enumerate(pred_scores):
            all_scores.append(score)

            if len(gt_bboxes) == 0 or len(matched_gts) == len(gt_bboxes):
                all_matches.append(0)  # FP (没目标了，或者目标全被匹配完了)
                continue

            # 提取当前预测框对所有 GT 的相似度
            similarities = safit_matrix[p_idx]
            best_gt_idx = np.argmax(similarities)
            best_safit = similarities[best_gt_idx]

            # 审判条件：相似度达标 且 该 GT 没被认领过
            if best_safit >= match_thresh and best_gt_idx not in matched_gts:
                all_matches.append(1)  # TP
                matched_gts.add(best_gt_idx)
            else:
                all_matches.append(0)  # FP

    if total_gts == 0 or len(all_scores) == 0:
        print("❌ 未发现任何目标或预测，跳过 F1 计算。")
        return

    print("📈 正在计算指标曲线...")
    # 3. 全局排序与指标计算
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)

    global_sort = np.argsort(all_scores)[::-1]
    all_scores = all_scores[global_sort]
    all_matches = all_matches[global_sort]

    acc_tp = np.cumsum(all_matches)
    acc_fp = np.cumsum(1 - all_matches)

    precision = acc_tp / (acc_tp + acc_fp + 1e-7)
    recall = acc_tp / (total_gts + 1e-7)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)

    # 4. 获取最佳操作点
    best_idx = np.argmax(f1_scores)
    best_thresh = all_scores[best_idx]
    best_f1 = f1_scores[best_idx]
    best_p = precision[best_idx]
    best_r = recall[best_idx]

    print("-" * 50)
    print(f"🏆 [Safit > {match_thresh}] 最佳部署配置:")
    print(f"推荐 score_thr (部署截断阈值): {best_thresh:.3f}")
    print(f"最高 F1-Score: {best_f1:.3f}")
    print(f"此时 Precision: {best_p:.3f} | Recall: {best_r:.3f}")
    print("-" * 50)

    # 5. 绘制并保存曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(all_scores, f1_scores, label='Safit F1-Score', color='green', linewidth=2)
    plt.plot(all_scores, precision, label='Precision', color='blue', linestyle='--')
    plt.plot(all_scores, recall, label='Recall', color='red', linestyle='--')

    plt.scatter(best_thresh, best_f1, color='gold', s=100, zorder=5, edgecolor='black',
                label=f'Best F1 ({best_f1:.2f} @ {best_thresh:.2f})')
    plt.axvline(x=best_thresh, color='gray', linestyle=':', alpha=0.7)

    plt.title(f'Detection Metrics vs. Confidence (Safit Match > {match_thresh})', fontsize=14)
    plt.xlabel('Confidence Threshold (score_thr)', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f'f1_curve_safit{match_thresh}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 曲线图已保存至: {save_path}\n")

def parse_args():  #cx 注意cocoeval已经换了！！
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    # parser.add_argument('--config',default='../qfdet_configs/qfdet_star_r50_fpn_1x_rgbtdroneperson_train01_debug.py' ,help='test config file path')
    # parser.add_argument('--checkpoint',default='workdir/cx01/epoch_12.pth' ,help='checkpoint file')
    # parser.add_argument('--out',default='/media/zhx/DATA/yxy/0827/mmdet-rgbtdroneperson-main/workdir/test01/out_01.pkl', help='output result file in pickle format')

    # 01
    parser.add_argument('--config',default='../qfdet_configs/qfdet_star_r50_fpn_1x_rgbtdroneperson_test01.py' ,help='test config file path')  #wr
    parser.add_argument('--checkpoint', default='../workdir/cx/20260604/epoch_5.pth', help='checkpoint file')  # wr
    # parser.add_argument('--checkpoint',default='../../../../results/T/epoch_12.pth' ,help='checkpoint file')   #wr
    parser.add_argument('--out',default='../workdir/test01/out_01.pkl', help='output result file in pickle format')  #wr
    #00
    # parser.add_argument('--config',default='../qfdet_configs/qfdet_star_r50_fpn_1x_rgbtdroneperson_test00_initial.py' ,help='test config file path')  #wr
    # parser.add_argument('--checkpoint',default='../../../../results/RGB/epoch_9.pth' ,help='checkpoint file')   #wr
    # parser.add_argument('--out',default='../workdir/test00/out_00.pkl', help='output result file in pickle format')  #wr


    # parser.add_argument(
    #     '--work-dir',
    #     default= '/media/zhx/DATA/yxy/0827/mmdet-rgbtdroneperson-main/workdir/test00/',
    #     help='the directory to save the file containing evaluation metrics')     #wr
    parser.add_argument(
        '--work-dir',
        default='../workdir/test01/',
        help='the directory to save the file containing evaluation metrics')  #wr
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        default=False,
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')  #cx 取消--
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default = 'bbox',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    # parser.add_argument(
    #     '--show-dir',default='/media/zhx/DATA/yxy/0827/mmdet-rgbtdroneperson-main/workdir/test/show_dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-dir',default=None, help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)') #可视化时过滤检测框的置信度阈值，默认 0.3
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)  #batch_size默认是1

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # init rfnext if 'RFSearchHook' is defined in cfg
    rfnext_init_model(model, cfg=cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    # if not os.path.exists(args.out):  #wr如果文夹存在，直接加载该文件，不再重新运行模型测试，造成无论什么样的权重，结果都一样
    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                args.show_score_thr)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)

        # In multi_gpu_test, if tmpdir is None, some tesnors
        # will init on cuda by default, and no device choice supported.
        # Init a tmpdir to avoid error on npu here.
        if cfg.device == 'npu' and args.tmpdir is None:
            args.tmpdir = './npu_tmpdir'

        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))
    # else:
    #     outputs = mmcv.load(args.out) #wr

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)  #COCO评估入口
            print(metric)
            metric_dict = dict(config=args.config, metric=metric) #这样可以将配置与结果关联起来，便于实验管理
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file) #使用 MMCV 的 dump 函数将 metric_dict 写入 json_file 文件

            # ==========================================================
            # 🆕 新增：在 COCO 评估完成后，自动执行工程视角的 F1 曲线评估
            # ==========================================================
            if rank == 0:  # 确保在分布式测试中只在主进程执行
                save_dir = args.work_dir if args.work_dir else './'
                # 默认评估 IoU=0.5 时的 F1-Score (工业界最常用的目标检测容忍度)
                compute_and_plot_safit_f1(dataset, outputs, save_dir, match_thresh=0.5, C_nwd=32.0)
            # ==========================================================


if __name__ == '__main__':
    main()
