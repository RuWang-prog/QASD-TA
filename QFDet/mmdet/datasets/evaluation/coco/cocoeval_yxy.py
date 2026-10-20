__author__ = 'tsungyi'

import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
import copy
import math

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds()) #将真实标注数据集（cocoGt）中的所有图像 ID 取出、排序后，赋值给评估参数对象（self.params）的 imgIds 属性。
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):  #用于将一组标注（anns）中的分割信息（多边形或 RLE）转换为二值掩码（binary mask），通常用于计算分割任务的 IoU 或进行可视化
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)) #加载指定图像和类别范围内的真实标注的典型调用
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)) #加载指定图像和类别范围内的预测标注
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            # gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            # gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']

            gt['ignore'] = 1 if (gt.get('ignore', 0) or gt.get('iscrowd', 0)) else 0  #wr20260326

            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))  #对 imgIds 和 catIds 进行去重并转为列表，确保后续迭代不会重复
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()  #对真实标注（ground truth）和检测结果（detection results）进行预处理和格式化，为后续的 IoU 计算和精度评估做准备
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU  #使用 computeIoU 方法计算边界框或掩膜之间的 IoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks #使用 computeOks（Object Keypoint Similarity）计算关键点相似度
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}  #计算所有图像-类别的 IoU 矩阵， computeIoU(imgId, catId) 返回一个二维数组 ious，维度为 [len(detections), len(annotations)]，其中每个元素是检测框与该图像中该类别的所有真实框之间的 IoU

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds #遍历每个类别 ID
                 for areaRng in p.areaRng #遍历每个面积范围
                 for imgId in p.imgIds #遍历每张图像 ID
             ]
        self._paramsEval = copy.deepcopy(self.params) #参数快照 _paramsEval 会在后续的 accumulate() 和 summarize() 中被引用，以保证指标计算的一致性
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats: #启用了按类别评估 (useCats=True)
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')  #用于获取检测结果按置信度降序排列的索引
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt] #提取 bbox 字段（格式为 [x, y, width, height]）
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        # iscrowd = [int(o['iscrowd']) for o in gt]
        # ious = maskUtils.iou(d,g,iscrowd)
        
        # d, g = np.array(d), np.array(g)
        # if g.shape[0] == 0 or d.shape[0] == 0:
        #     return []
                # d, g = self.xyhw2xxyy(d), self.xyhw2xxyy(g)
        # ious = self.iou_batch(d,g)
        # ious = self.dis_batch1(d,g)
        # ious = self.NWD(d,g,C=16)
        # ious = self.dis_IOU_batch(d,g)
        
        # my metric 
        d, g = np.array(d), np.array(g)
        if g.shape[0] == 0 or d.shape[0] == 0:
            return []
        # ious = self.NWD_IOU_batch(d,g,C=32) #调用 self.NWD_IOU_batch(d, g, C=32) 计算相似度矩阵  safit
        # ious = self.QANWD_IOU_batch(d, g, C=16.0)  #wr20260529
        # ious = self.iou_batch(d, g)
        ious = self.QAsafit_batch(d,g)  #wr20260903

        return ious
    
    def xyhw2xxyy(self, bbox):
        output = bbox.copy()
        output[:,0] = bbox[:,0]
        output[:,1] = bbox[:,1]
        output[:,2] = bbox[:,0] + bbox[:,2]
        output[:,3] = bbox[:,1] + bbox[:,3]
        return output
    
    def iou_batch(self, bb_test, bb_gt):
        """
        From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
        """
        # print(bb_test.shape)
        # print(bb_gt.shape)
        bb_test, bb_gt = self.xyhw2xxyy(bb_test), self.xyhw2xxyy(bb_gt) #将边界框表示从 [x, y, w, h]（左上角坐标 + 宽高）转换为 [x1, y1, x2, y2]（左上角 + 右下角坐标）
        bb_gt = np.expand_dims(bb_gt, 0) #维度扩展以利用广播机制 形状变为 (1, N, 4)
        bb_test = np.expand_dims(bb_test, 1) #形状变为 (M, 1, 4)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1) #如果两个框不相交，则 xx2 - xx1 或 yy2 - yy1 将为负数
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh + 1e-7) #分母：检测框面积 + 真实框面积 - 交集面积。 o 即为 IoU 矩阵，值域在 [0, 1]
        return (o) #在 Python 中，return (o) 和 return o 是完全等价的 括号是合法的但非必需的
    
    # def NWD(self, bb_test, bb_gt, C):
    #     bb_test[:,2:4], bb_gt[:,2:4] = bb_test[:,2:4]/2, bb_gt[:,2:4]/2  #输入边界框的格式为 [x, y, w, h] 得到半宽和半高
    #     bb_gt = np.expand_dims(bb_gt, 0) # (1,b,4)
    #     bb_test = np.expand_dims(bb_test, 1) # (b,1,4)
    #     return np.exp(-np.sqrt(np.sum((bb_test - bb_gt)*(bb_test - bb_gt), axis=2))/C) #计算指数归一化的相似度 二者相减并平方后，得到形状为 (M, N, 4) 的张量，其中最后一个维度（索引 2）对应边界框的 4 个坐标分量（[x, y, w/2, h/2]）。对 axis=2 求和，即对每个测试框-真实框对的 4 个坐标分量的平方进行累加，得到形状为 (M, N) 的距离平方矩阵。

    #wr20260406
    def NWD(self, bb_test, bb_gt, C):
        """
        计算基于欧氏距离的 NWD 相似度（适用于小目标）
        输入: bb_test, bb_gt: 形状为 (N,4) 和 (M,4)，格式为 [x_min, y_min, width, height]
        输出: 形状为 (N, M) 的相似度矩阵，值域 (0,1]
        """
        # 避免修改原始数组
        bb_test = bb_test.copy()
        bb_gt = bb_gt.copy()

        # 1. 将左上角 (x_min, y_min) 转换为中心点 (cx, cy)
        cx_test = bb_test[:, 0] + bb_test[:, 2] / 2
        cy_test = bb_test[:, 1] + bb_test[:, 3] / 2
        cx_gt = bb_gt[:, 0] + bb_gt[:, 2] / 2
        cy_gt = bb_gt[:, 1] + bb_gt[:, 3] / 2

        # 2. 计算半宽和半高
        half_w_test = bb_test[:, 2] / 2
        half_h_test = bb_test[:, 3] / 2
        half_w_gt = bb_gt[:, 2] / 2
        half_h_gt = bb_gt[:, 3] / 2

        # 3. 组成四维向量 [cx, cy, w/2, h/2]
        vec_test = np.stack([cx_test, cy_test, half_w_test, half_h_test], axis=1)
        vec_gt = np.stack([cx_gt, cy_gt, half_w_gt, half_h_gt], axis=1)

        # 4. 扩展维度以计算所有配对之间的欧氏距离
        vec_gt_exp = np.expand_dims(vec_gt, 0)  # (1, M, 4)
        vec_test_exp = np.expand_dims(vec_test, 1)  # (N, 1, 4)

        diff = vec_test_exp - vec_gt_exp  # (N, M, 4)
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N, M)

        # 5. 高斯核映射到相似度
        return np.exp(-dist / C)

    def dis_batch1(self, bb_test, bb_gt):
        bb_test, bb_gt = self.xyhw2xxyy(bb_test), self.xyhw2xxyy(bb_gt)
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx0 = np.minimum(bb_test[..., 0], bb_gt[..., 0])
        yy0 = np.minimum(bb_test[..., 1], bb_gt[..., 1])
        xx1 = np.maximum(bb_test[..., 2], bb_gt[..., 2])
        yy1 = np.maximum(bb_test[..., 3], bb_gt[..., 3])
        dis_long = (xx1 - xx0) ** 2 + (yy1-yy0) ** 2
        
        dis_centroid = (((bb_test[..., 0] + bb_test[..., 2]) - (bb_gt[..., 0] + bb_gt[..., 2])) ** 2 + ((bb_test[..., 1] + bb_test[..., 3]) - (bb_gt[..., 1] + bb_gt[..., 3])) ** 2)/4
        return 1 - dis_centroid / dis_long
    
    def dis_IOU_batch(self, bb_test, bb_gt):
        bb_test1, bb_gt1 = bb_test.copy(), bb_gt.copy()
        bb_test2, bb_gt2 = bb_test.copy(), bb_gt.copy()
        bb_gt = np.expand_dims(bb_gt, 0)
        Area_gt = (bb_gt[..., 2]*bb_gt[..., 3]).repeat(bb_test.shape[0], axis=0)
        Area_gt = np.abs(Area_gt)
        C = 32 ** 2 / (-math.log(0.5, math.e))**2
        # print((self.dis_batch1(bb_test, bb_gt)).shape)
        # print((self.iou_batch(bb_test, bb_gt)).shape)
        # print((np.exp(-np.sqrt(Area_gt/C))*(self.dis_batch1(bb_test, bb_gt))+(1-np.exp(-np.sqrt(Area_gt/C)))*self.iou_batch(bb_test, bb_gt)).shape)
        iou = self.iou_batch(bb_test1, bb_gt1)
        dis = self.dis_batch1(bb_test2, bb_gt2)
        
        ious = np.exp(-np.sqrt(Area_gt/C))*(dis)+(1-np.exp(-np.sqrt(Area_gt/C)))*iou
        
        return ious
    
    # def dis_IOU_batch(self, bb_test, bb_gt):
    #     bb_gt = np.expand_dims(bb_gt, 0)
    #     Area_gt = ((bb_gt[..., 2]-bb_gt[..., 0])*(bb_gt[..., 3]-bb_gt[..., 1])).repeat(bb_test.shape[0], axis=0)
    #     C = 32 ** 2
        
    #     # iou = self.iou_batch(bb_test, bb_gt)
    #     # dis = self.dis_batch1(bb_test, bb_gt)
    #     # print(iou.shape)
    #     # print(dis.shape)
    #     # ious = np.exp(-np.sqrt(Area_gt/C))*(dis)+(1-np.exp(-np.sqrt(Area_gt/C)))*iou
    #     ious = np.exp(-np.sqrt(Area_gt/C))*(self.dis_batch1(bb_test, bb_gt))+(1-np.exp(-np.sqrt(Area_gt/C)))*self.iou_batch(bb_test, bb_gt)
        
    #     return ious[0,:,:]
    
    def sigmoid(self, x, k):
        return 1/(1+np.exp(-(1/k) * x))

    # initial safit
    def NWD_IOU_batch(self, bb_test, bb_gt, C):
        bb_test1, bb_gt1 = bb_test.copy(), bb_gt.copy()
        bb_test2, bb_gt2 = bb_test.copy(), bb_gt.copy()
        bb_gt = np.expand_dims(bb_gt, 0)
        Area_gt = ((bb_gt[..., 2])*(bb_gt[..., 3])).repeat(bb_test.shape[0], axis=0)
        Area_gt = np.abs(Area_gt) #将变量 Area_gt 中的每个元素取绝对值，并将结果重新赋值给 Area_gt
        # print(Area_gt.shape)
        iou = self.iou_batch(bb_test1, bb_gt1) #计算 IoU 矩阵
        nwd = self.NWD(bb_test2, bb_gt2, C) #计算 NWD 矩阵  bb_test2,bb_gt2(x,y,w,h)左上角+宽高
        # print(iou.shape)
        # print(nwd.shape)
        ious = (self.sigmoid(np.sqrt(Area_gt)-C, C)) * iou+\
            (1-self.sigmoid(np.sqrt(Area_gt)-C, C)) * nwd
        # print(ious>0.5)
        # print(iou)
        # print(nwd)
        return ious

    #wr20260605 QAsafit

    def QAsafit_batch(self, bb_test, bb_gt, C_min=16.0, boundary=32.0, tau=4.0):
        """
        全尺度融合评估矩阵 QAsafit (IoU-centric 主视角)
        :param bb_test: 预测框数组，Shape: [M, 4]，格式必须为 [x_topleft, y_topleft, w, h]
        :param bb_gt: 真实框数组，Shape: [N, 4]，格式必须为 [x_topleft, y_topleft, w, h]
        :return: M x N 的融合相似度矩阵
        """
        # 防止原地修改传入的数组
        bb_test1, bb_gt1 = bb_test.copy(), bb_gt.copy()

        # 1. 提取 GT 的宽(w)和高(h)，并计算真实目标尺度 (sqrtArea)
        # bb_gt[:, 2] 是 width, bb_gt[:, 3] 是 height
        bb_gt_exp = np.expand_dims(bb_gt, 0)  # Shape: [1, N, 4]
        Area_gt = (bb_gt_exp[..., 2] * bb_gt_exp[..., 3]).repeat(bb_test.shape[0], axis=0)
        Area_gt = np.abs(Area_gt)
        sqrtA = np.sqrt(Area_gt)  # Shape: [M, N]

        # 2. 获取原生的 IoU 矩阵
        # ⚠️注意: 请确保你自己的 self.iou_batch 函数也支持传入 xywh 格式！
        # 如果你的 iou_batch 需要 xyxy (左上角, 右下角)，你需要在此处做一次转换。
        iou_matrix = self.iou_batch(bb_test1, bb_gt1)

        # ==========================================
        # 3. 向量化计算 QA-NWD 矩阵
        # ==========================================
        # 计算中心点坐标 (中心点 = 左上角坐标 + 宽高的一半)
        dt_cx = bb_test[:, 0] + bb_test[:, 2] / 2.0
        dt_cy = bb_test[:, 1] + bb_test[:, 3] / 2.0
        gt_cx = bb_gt[:, 0] + bb_gt[:, 2] / 2.0
        gt_cy = bb_gt[:, 1] + bb_gt[:, 3] / 2.0

        # 提取宽高进行惩罚计算
        dt_w, dt_h = bb_test[:, 2], bb_test[:, 3]
        gt_w, gt_h = bb_gt[:, 2], bb_gt[:, 3]

        # 利用 NumPy 广播机制计算差值矩阵 (Shape: [M, N])
        dx = dt_cx[:, None] - gt_cx[None, :]
        dy = dt_cy[:, None] - gt_cy[None, :]
        dw = dt_w[:, None] - gt_w[None, :]
        dh = dt_h[:, None] - gt_h[None, :]

        # 计算二维高斯分布的 Wasserstein 距离 (W2)
        w2_sq = dx ** 2 + dy ** 2 + (dw ** 2 + dh ** 2) / 4.0
        w2 = np.sqrt(np.maximum(w2_sq, 1e-12))

        # 动态分母：基于 GT 尺度，底线为 C_min (16.0)
        C_dyn = np.maximum(sqrtA, C_min)
        qa_nwd_matrix = np.exp(-w2 / C_dyn)

        # ==========================================
        # 4. 尺度路由与融合 (IoU 主视角)
        # ==========================================
        # 计算 IoU 的统治权重。
        # 负号 -(sqrtA - boundary) 保证：大目标 w_iou -> 1.0，小目标 w_iou -> 0.0
        w_iou = 1.0 / (1.0 + np.exp(-(sqrtA - boundary) / tau))

        # 融合：由 IoU 主导，QA-NWD 在小目标区间作为补充
        qasafit_matrix = w_iou * iou_matrix + (1.0 - w_iou) * qa_nwd_matrix



        return qasafit_matrix

    # #wr20260509 safit
    # def NWD_IOU_batch(self, bb_test, bb_gt, C):
    #     bb_test1, bb_gt1 = bb_test.copy(), bb_gt.copy()
    #     bb_test2, bb_gt2 = bb_test.copy(), bb_gt.copy()
    #     bb_gt = np.expand_dims(bb_gt, 0)
    #     Area_gt = ((bb_gt[..., 2]) * (bb_gt[..., 3])).repeat(bb_test.shape[0], axis=0)
    #     Area_gt = np.abs(Area_gt)
    #
    #     # 1. 单独保留原始 IoU 矩阵，用于后续的硬性判断
    #     iou_matrix = self.iou_batch(bb_test1, bb_gt1)
    #
    #     # 2. 计算 NWD 矩阵
    #     nwd_matrix = self.NWD(bb_test2, bb_gt2, C)
    #
    #     # 3. 计算融合尺度权重的 Safit 相似度矩阵
    #     w = self.sigmoid(np.sqrt(Area_gt) - C, C)
    #     safit_matrix = w * iou_matrix + (1 - w) * nwd_matrix
    #
    #
    #
    #     return safit_matrix

    #wr 20260529 QA-NWD
    def QANWD_IOU_batch(self, bb_test, bb_gt, C=16.0):
        """
        将原有的 Safit 融合矩阵完全重构为纯粹的 QA-NWD 相似度矩阵
        :param bb_test: 预测框矩阵, 形状 (M, 4), 格式 [x, y, w, h]
        :param bb_gt: 真实框矩阵, 形状 (N, 4), 格式 [x, y, w, h]
        :param C: 此处作为 C_min (微观目标的保底物理容忍度), 全尺度推荐设为 16.0
        :return: 形状为 (M, N) 的 QA-NWD 相似度矩阵
        """
        M, N = bb_test.shape[0], bb_gt.shape[0]
        if M == 0 or N == 0:
            return np.zeros((M, N), dtype=np.float32)

        # 确保数据格式为 float32
        dt = bb_test.astype(np.float32)
        gt = bb_gt.astype(np.float32)

        # 1. 提取并计算 dt 和 gt 的中心点 (cx, cy)
        dt_cx = dt[:, 0] + dt[:, 2] / 2.0
        dt_cy = dt[:, 1] + dt[:, 3] / 2.0
        dt_w = dt[:, 2]
        dt_h = dt[:, 3]

        gt_cx = gt[:, 0] + gt[:, 2] / 2.0
        gt_cy = gt[:, 1] + gt[:, 3] / 2.0
        gt_w = gt[:, 2]
        gt_h = gt[:, 3]

        # 2. 利用 NumPy 广播机制计算差值矩阵 (Shape: M x N)
        dx = dt_cx[:, None] - gt_cx[None, :]
        dy = dt_cy[:, None] - gt_cy[None, :]
        dw = dt_w[:, None] - gt_w[None, :]
        dh = dt_h[:, None] - gt_h[None, :]

        # 3. 计算 2nd Wasserstein 距离 (W2)
        w2_sq = dx ** 2 + dy ** 2 + (dw ** 2 + dh ** 2) / 4.0
        # 加上微小的 epsilon 防止对 0 开根号导致梯度或数值异常
        w2 = np.sqrt(np.maximum(w2_sq, 1e-12))

        # 4. 计算动态自适应分母 C_dyn = max(sqrt(Area_gt), C_min)
        # 注意: 按照 QA-NWD 理论，评估标尺应当绑定真实目标的物理尺度
        gt_area = np.abs(gt_w * gt_h)
        gt_sqrt_area = np.sqrt(gt_area)

        # C 在此处作为保底红线 C_min，阻止微观目标的量化误差引起的得分雪崩
        C_dyn = np.maximum(gt_sqrt_area, C)  # Shape: (N,)
        C_dyn_matrix = C_dyn[None, :]  # 广播至 Shape: (1, N) 适配距离矩阵

        # 5. 计算终极 QA-NWD 相似度分数 (值域严格在 (0, 1] 空间)
        qa_nwd_matrix = np.exp(-w2 / C_dyn_matrix)

        return qa_nwd_matrix


    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]): #每个真值可能原本就有 ignore 标记 再根据当前面积范围 aRng 判断：如果真值面积不在范围内，也视为忽略
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort') #对真实标注列表 gt 按照忽略标志 _ignore 进行稳定排序，并返回排序后的索引。
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort') #检测按分数降序排序（负分实现降序） mergesort 保证稳定排序，即相同分数时保持原始顺序
        dt = [dt[i] for i in dtind[0:maxDet]] #并只保留前 maxDet 个
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId] #由于真值已重新排序（gtind），需要对 IoU 矩阵的列按 gtind 重新排列，使列顺序与排序后的 gt 一致

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))  #gtm 和 dtm 将记录在每个 IoU 阈值下，每个真值/检测匹配到的对方 ID
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0: #核心匹配算法（逐 IoU 阈值）
            for tind, t in enumerate(p.iouThrs):  #tind是阈值的索引
                for dind, d in enumerate(dt):  #按分数降序遍历所有检测结果（dt）
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])  #  1-1e-10 0.9999999999 1 减去 10 的负 10 次方（0.0000000001）
                    m   = -1  #为每个检测记录当前最佳匹配的真实框索引 m（初始 -1）和当前最大 IoU iou（初始为该阈值，即 t，但上限为 1-1e-10
                    for gind, g in enumerate(gt):  #遍历所有真实框（gt） gind是gt的索引
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1: #如果已经找到了一个匹配（m > -1），并且该匹配的真实框是非忽略的（gtIg[m]==0），而当前真实框是忽略的（gtIg[gind]==1），则提前终止（break）。这是为了优先匹配非忽略框，若已经匹配到一个非忽略框，后续忽略框不再考虑（因为检测只应匹配一个真实框，而忽略框优先级低）
                            break
                        # continue to next gt unless better match made
                        try:
                            if ious[dind,gind] < iou: #如果当前检测与当前真实框的 IoU（ious[dind, gind]）小于当前最佳 IoU，则跳过该真实框
                                continue
                        except:
                            print(ious)
                            print(ious[dind,gind])
                            print(iou)
                            if ious[dind,gind] < iou:
                                continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]  #否则，更新最佳匹配为当前真实框，并更新最大 IoU。
                        m=gind
                    # if match made， store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]  # dtIg[tind, dind] 记录了该检测是否应被忽略（等于其匹配真值的忽略标记）
                    dtm[tind,dind]  = gt[m]['id'] # dtm[tind, dind] 记录了检测匹配到的真值 ID
                    gtm[tind,m]     = d['id'] # gtm[tind, m] 记录了真值匹配到的检测 ID
        # set unmatched detections outside of area range to ignore  将区域范围之外的不匹配检测结果设置为忽略
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0))) #完善检测框的忽略标记 np.logical_and(dtm==0, np.repeat(a, T, 0))：选出那些既未匹配任何真实框，又全局应被忽略的检测框，标记为 True
        # store results for given image and category 返回结果字典
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt], # 检测的 ID 列表（按排序后）
                'gtIds':        [g['id'] for g in gt], # 真值的 ID 列表（按排序后）
                'dtMatches':    dtm, # (T, D) 检测匹配的真值 ID
                'gtMatches':    gtm, # (T, G) 真值匹配的检测 ID
                'dtScores':     [d['score'] for d in dt], # 检测分数（排序后）
                'gtIgnore':     gtIg,  # (G,) 每个真值是否忽略
                'dtIgnore':     dtIg,   # (T, D) 每个检测是否忽略（按阈值）
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)  #IoU 阈值个数
        R           = len(p.recThrs)  #召回率阈值个数
        K           = len(p.catIds) if p.useCats else 1 #类别数
        A           = len(p.areaRng) #面积范围个数
        M           = len(p.maxDets) #最大检测数个数
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories 插值精度
        recall      = -np.ones((T,K,A,M)) #最大召回率（即所有检测点中的最后一个召回率）
        scores      = -np.ones((T,R,K,A,M)) #达到该插值精度时对应的检测分数阈值（用于绘制 PR 曲线时使用，但通常不直接输出）

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections  检索每个类别、区域范围和最大检测次数下的 E 值； 用于按类别、IoU 阈值、面积范围和最大检测数，收集所有图像的检测分数，以便后续计算 Precision‑Recall 曲线
        for k, k0 in enumerate(k_list): #全部类别
            Nk = k0*A0*I0  # 该类别在 evalImgs 中的起始偏移（假设类别优先）
            for a, a0 in enumerate(a_list): #面积范围
                Na = a0*I0   # 该面积范围在 evalImgs 中的起始偏移（相对于当前类别的偏移）
                for m, maxDet in enumerate(m_list): #所有最大检测数选项
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]  # 过滤掉没有结果（无检测或评估未完成）的图像
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E]) # 从每个有效结果中取出该图像所有检测框的分数，只保留前 maxDet 个（已排序），然后拼接

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort') #将检测分数 dtScores 按从高到低的顺序排序，并返回排序后元素在原始数组中的索引。
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds] #将所有图像的匹配结果按检测分数全局排序后合并。
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )   #统计所有非忽略的真实标注（ground truth）的数量
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) ) #计算 TP 和 FP 累计和 真正例 ；标记每个检测框是否为真阳性（True Positive, TP）； 当 dtm[t, d] != 0（即该检测匹配到了真实框） 且 dtIg[t, d] == False（即该检测未被忽略）时，结果为 True，表示该检测是真阳性（TP）
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) ) #假正例 ；标记每个检测框是否为假阳性（False Positive, FP）。该检测未匹配任何真实标注且没有被忽略。这正是假阳性（FP）的定义

                    # tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float) #wr  沿检测顺序（列方向）计算累计和，得到在每个排序位置上的累计 TP 和 FP 数量
                    # fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)   #wr
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):  #遍历每个 IoU 阈值，计算召回率、精度并插值 tp/fp：来自 tp_sum /fp_sum的一行，是一个一维数组，形状为 (24814,)。
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig   #计算每个排序位置的召回率

                        pr = tp / (fp+tp+np.spacing(1))  #计算每个排序位置的精度  np.spacing(1) 1的浮点精度步长，防止分母为零
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]  #记录该 IoU 阈值下的最大召回率 rc[-1]（即所有检测都考虑时的召回率）存入 recall[t,k,a,m]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement  对精度序列进行单调化处理，确保在召回率增加时精度不会出现局部上升，这是计算平均精度（AP）的标准步骤之一。
                        pr = pr.tolist(); q = q.tolist() #将 NumPy 数组 pr 和 q 转换为 Python 内置的列表对象

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')  #在召回率曲线上查找指定召回率阈值位置的常用操作；inds 给出了每个阈值在 rc 中的索引，然后可以用 pr[inds] 取出对应位置的精度值（若 inds[j] == len(rc)，则取 0）。
                        try:
                            for ri, pi in enumerate(inds): #在指定召回率阈值上采样精度和分数的关键步骤，用于后续计算平均精度（AP）以及记录对应的分数阈值。
                                q[ri] = pr[pi] #存储在每个预定义召回率阈值上插值得到的精度
                                ss[ri] = dtScoresSorted[pi] #存储每个召回率阈值对应的分数阈值
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)  #将插值结果存入 precision[t,:,k,a,m] 和 scores[t,:,k,a,m]
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }  #存储最终结果
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets(): #检测任务汇总函数 _summarizeDets
            stats = np.zeros((16,))
            stats[0] = _summarize(1) #AP 相关（ap=1） 默认 AP（IoU=0.50:0.95，area='all'，maxDets=100）
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='extreme tiny', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='tiny', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[7] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[0]) #AR 相关（ap=0）
            stats[9] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[10] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='extreme tiny', maxDets=self.params.maxDets[2])
            stats[12] = _summarize(0, areaRng='tiny', maxDets=self.params.maxDets[2])
            stats[13] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[14] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[15] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps(): #关键点任务汇总函数 _summarizeKps
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType  #根据 iouType 选择对应的汇总函数：分割/检测任务用 _summarizeDets，关键点用 _summarizeKps
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 8 ** 2], [8 ** 2, 16 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'extreme tiny', 'tiny', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 8 ** 2], [8 ** 2, 16 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'extreme tiny', 'tiny', 'small', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
