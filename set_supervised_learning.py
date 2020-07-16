import os
import torch

from detectron2 import model_zoo
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, verify_results

from setrep import register_data, add_setrep_config, ClevrStateEvaluator, DatasetMapper


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if cfg.MODEL.FLOWSETNET.TARGET == 'BOX':
            evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif cfg.MODEL.FLOWSETNET.TARGET == 'STATE':
            evaluator = ClevrStateEvaluator(dataset_name, True, output_folder)
        return evaluator

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    @classmethod
    def build_optimizer(cls, cfg, model):
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=cfg.SOLVER.BASE_LR
        )
        return optimizer


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    # setup
    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    add_setrep_config(cfg)
    cfg.merge_from_file("/home/fengc/PlayGround/object_detection/SetRep/config/Base-FlowSetNet.yaml")
    register_data('tcoco')
    cfg.DATASETS.TRAIN = ("tcoco_train",)
    cfg.DATASETS.TEST = ("tcoco_val",)
    cfg.OUTPUT_DIR = 'test/'
    # torch.backends.cudnn.benchmark = True

    # eval
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
