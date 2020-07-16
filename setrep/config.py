# -*- coding = utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_setrep_config(cfg):
    """
    Add config for densepose head.
    """
    _C = cfg

    _C.MODEL.FLOWSETNET = CN()
    _C.MODEL.FLOWSETNET.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    _C.MODEL.FLOWSETNET.DECODER_TYPE = 'MLP'
    _C.MODEL.FLOWSETNET.GNN_ITERATION = 5
    _C.MODEL.FLOWSETNET.EMBEDDING_TYPE = 'NO_EDGE'
    _C.MODEL.FLOWSETNET.TARGET = 'BOX'
    _C.MODEL.FLOWSETNET.ELEMENT_DIM = 4
    _C.MODEL.FLOWSETNET.MAX_ELEMENT_NUM = 10
    _C.MODEL.FLOWSETNET.CODE_CHANNELS = 512
    _C.MODEL.FLOWSETNET.LOSS_FUNCTION = 'WITH_BCE'
    _C.MODEL.FLOWSETNET.SMOOTH_L1_BETA = 0.0
    _C.MODEL.FLOWSETNET.SCORE_THRESH_TEST = 0.5
