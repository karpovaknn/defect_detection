# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _summarize(cocoEval, ap=1, iouThr=None, areaRng='all', maxDets=100 ):
    p = cocoEval.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap==1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = cocoEval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,:,aind,mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = cocoEval.eval['recall']
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


def analyze_results(res_file,
                    ann_file,
                    areas=None):
    if areas:
        assert len(areas) == 3, '3 integers should be specified as areas, \
            representing 3 area regions'

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = cocoGt.getImgIds()

    iou_type = 'bbox'
    cocoEval = COCOeval(
        copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), iou_type)
    cocoEval.params.imgIds = imgIds
    if areas:
        cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                   [areas[0], areas[1]],
                                   [areas[1], areas[2]]]
    cocoEval.evaluate()
    cocoEval.accumulate()
    stats = np.zeros((10,))
    stats[0] = _summarize(cocoEval, 1)
    stats[1] = _summarize(cocoEval, 1, iouThr=.5, maxDets=cocoEval.params.maxDets[2])
    stats[2] = _summarize(cocoEval, 1, iouThr=.5, areaRng='small', maxDets=cocoEval.params.maxDets[2])
    stats[3] = _summarize(cocoEval, 1, iouThr=.5, areaRng='medium', maxDets=cocoEval.params.maxDets[2])
    stats[4] = _summarize(cocoEval, 1, iouThr=.5,areaRng='large', maxDets=cocoEval.params.maxDets[2])
    stats[5] = _summarize(cocoEval, 0, maxDets=cocoEval.params.maxDets[2])
    stats[6] = _summarize(cocoEval, 0, iouThr=.5, maxDets=cocoEval.params.maxDets[2])
    stats[7] = _summarize(cocoEval, 0, iouThr=.5, areaRng='small', maxDets=cocoEval.params.maxDets[2])
    stats[8] = _summarize(cocoEval, 0, iouThr=.5, areaRng='medium', maxDets=cocoEval.params.maxDets[2])
    stats[9] = _summarize(cocoEval, 0, iouThr=.5, areaRng='large', maxDets=cocoEval.params.maxDets[2])
        


def main():
    parser = ArgumentParser(description='COCO Error Analysis Tool')
    parser.add_argument('result', help='result file (json format) path')
    parser.add_argument(
        '--ann',
        default='data/coco/annotations/instances_val2017.json',
        help='annotation file path')
    parser.add_argument(
        '--areas',
        type=int,
        nargs='+',
        default=[1024, 9216, 10000000000],
        help='area regions')
    args = parser.parse_args()
    analyze_results(
        args.result,
        args.ann,
        areas=args.areas)


if __name__ == '__main__':
    main()


