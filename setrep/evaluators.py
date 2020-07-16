import os
import itertools
import copy
import numpy as np
from scipy.spatial.distance import cdist
import torch
from collections import OrderedDict, defaultdict
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import DatasetCatalog


def pair_wise_metric(g, d):
    g_coord = 3*np.array([i[:3] for i in g])
    d_coord = 3*np.array([i[:3] for i in d])
    dist = cdist(d_coord, g_coord)
    g_state = np.array([[np.argmax(i[3:5]), np.argmax(i[5:13]), np.argmax(i[13:16]), np.argmax(i[16:])] for i in g])[None,:,:]
    d_state = np.array([[np.argmax(i[3:5]), np.argmax(i[5:13]), np.argmax(i[13:16]), np.argmax(i[16:])] for i in d])[:,None,:]
    diff = np.logical_not((g_state - d_state).astype(np.bool).sum(-1)).astype(np.float)
    diff[diff<1.0] = np.inf
    metric = dist*diff
    return metric


def instances_to_dict(instances):
    num_instance = len(instances)
    if num_instance == 0:
        return []

    state = instances.state.numpy().tolist()
    scores = instances.scores.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "state": state[k],
            "score": scores[k],
        }
        results.append(result)
    return results


class ClevrStateEvaluator(DatasetEvaluator):
    """Evaluation Clevr State prediction"""

    def __init__(self, dataset_name, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

        self._gt = DatasetCatalog.get(dataset_name)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            instances = output["instances"].to(self._cpu_device)
            prediction["annotations"] = instances_to_dict(instances)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        state_eval = stateval(self._gt, self._predictions)
        state_eval.evaluate()
        state_eval.accumulate()
        state_eval.summarize()
        self._results['state'] = state_eval.stats


class stateval:
    def __init__(self, gt=None, dt=None, iouType='state'):
        self._gt = gt
        self._dt = dt
        self.imgIds = list(img['image_id'] for img in self._gt)
        self.evalImgs = defaultdict(list)   # per-image evaluation results
        self.eval     = {}                  # accumulated evaluation results
        self.distThrs = [0.125, 0.25, 0.5, 1.0, 'inf']
        self.stats = {}                     # result summarization
        self.dist = {}                      # ious between all gts and dts

    def _prepare(self):
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        for gt in self._gt:
            for obj in gt['annotations']:
                self._gts[gt['image_id']].append(obj)
        for dt in self._dt:
            for obj in dt['annotations']:
                self._dts[dt['image_id']].append(obj)

    def evaluate(self):
        self._prepare()
        self.dist = {imgId: self.computeMetric(imgId) for imgId in self.imgIds}
        self.evalImgs = [self.evaluateImg(imgId) for imgId in self.imgIds]

    def computeMetric(self, imgId):
        gt = self._gts[imgId]
        dt = self._dts[imgId]
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        g = [g['state'] for g in gt]
        d = [d['state'] for d in dt]
        dist = pair_wise_metric(g, d)

        return dist

    def evaluateImg(self, imgId):
        gt = self._gts[imgId]
        dt = self._dts[imgId]
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        dist = self.dist[imgId]
        T = len(self.distThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        if not len(dist)==0:
            for tind, t in enumerate(self.distThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = t
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0:
                            continue
                        # continue to next gt unless better match made
                        if iou != 'inf':
                            if dist[dind,gind] > iou:
                                continue
                        else:
                            if np.isinf(dist[dind,gind]):
                                continue
                        # if match successful and best so far, store appropriately
                        iou=dist[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtm[tind,dind]  = 1.0
                    gtm[tind,m]     = 1.0
        # store results for given image
        return {
                'image_id':     imgId,
                'dtMatches':    dtm,
                'gtNum':        G,
                'dtScores':     [d['score'] for d in dt],
            }

    def accumulate(self):
        dtScores = np.concatenate([e['dtScores'] for e in self.evalImgs])
        npo = np.array([e['gtNum'] for e in self.evalImgs]).sum()
        inds = np.argsort(-dtScores, kind='mergesort')
        dtm = np.concatenate([e['dtMatches'] for e in self.evalImgs], axis=1)[:, inds]
        acc_tp = np.cumsum(dtm, axis=1).astype(dtype=np.float)
        acc_fp = np.cumsum(np.logical_not(dtm), axis=1).astype(dtype=np.float)
        rc = acc_tp/npo
        pr = acc_tp/(acc_tp + acc_fp + np.spacing(1))
        self.eval = {
            'precision': pr,
            'recall':    rc,
        }

    def summarize(self):
        pr = self.eval['precision']
        rc = self.eval['recall']
        prp = np.concatenate((np.zeros((pr.shape[0], 1)), pr), -1)
        prr = np.concatenate((pr, np.zeros((pr.shape[0], 1))), -1)
        dif = prr-prp > 0
        prp[dif] = prr[dif]
        pr = prp[:,1:]
        rcr = np.concatenate((np.zeros((rc.shape[0], 1)), rc), -1)
        rcc = np.concatenate((rc, np.zeros((rc.shape[0], 1))), -1)
        res = (rcc-rcr)[:,1:-1]
        auc = (pr[:,1:]*res).sum(-1)
        for _, (t, ap) in enumerate(zip(self.distThrs, auc)):
            self.stats[str(t)] = ap
