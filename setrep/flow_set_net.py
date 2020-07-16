# Chao Feng
from typing import List
import numpy as np
import scipy.optimize
import torch
from torch import nn
from torch.nn import init
from fvcore.nn import smooth_l1_loss

from detectron2.structures import ImageList
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances
from detectron2.modeling import build_backbone
from detectron2.modeling import META_ARCH_REGISTRY

__all__ = ["FlowSetNet", "build_flowsetnet"]


def hungarian_loss(target, prediction, mask, beta):
    size = target.size()
    expanding_size = size[:2] + (size[1], size[2])
    target_set = target.unsqueeze(-3).expand(expanding_size)
    predict_set = prediction.unsqueeze(-2).expand(expanding_size)
    pair_dist = smooth_l1_loss(predict_set, target_set, beta).mean(-1)
    # pair_dist = pair_dist * mask.unsqueeze(-2)
    # calculating linear assignment cost using scipy.optimize.linear_sum_assignment, can only do it on cpu
    pair_dist_np = pair_dist.detach().cpu().numpy()
    indices = [hungarian_loss_per_sample(sample_pair_dist) for sample_pair_dist in pair_dist_np]
    losses = [
        sample[row_idx, col_idx].mean()
        for sample, (row_idx, col_idx) in zip(pair_dist, indices)
    ]
    total_loss = torch.mean(torch.stack(losses))
    return indices, total_loss

def hungarian_loss_per_sample(sample_np):
    return scipy.optimize.linear_sum_assignment(sample_np)


class FlowSetNet(nn.Module):
    """
    Flow-based Set predictor
    """

    def __init__(self, cfg, backbone, decoder):
        super().__init__()

        self.in_features = cfg.MODEL.FLOWSETNET.IN_FEATURES
        self.max_num = cfg.MODEL.FLOWSETNET.MAX_ELEMENT_NUM
        self.smooth_l1_beta = cfg.MODEL.FLOWSETNET.SMOOTH_L1_BETA
        self.score_thresh_test = cfg.MODEL.FLOWSETNET.SCORE_THRESH_TEST
        self.target = cfg.MODEL.FLOWSETNET.TARGET

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = backbone
        self.decoder = decoder
        self.input_format = cfg.INPUT.FORMAT
        self.mask_loss = nn.BCELoss(reduction = 'mean')

        for m in self.modules():
            if (
                    isinstance(m, nn.Linear)
                    or isinstance(m, nn.Conv2d)
                    or isinstance(m, nn.Conv1d)
            ):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        if cfg.MODEL.FLOWSETNET.LOSS_FUNCTION == 'WITH_BCE':
            self.losses = self.losses_with_bce
        elif cfg.MODEL.FLOWSETNET.LOSS_FUNCTION == 'EUCLIDEAN':
            self.losses = self.losses_eu

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        out_elements, mask_logits = self.decoder(features)

        image_shape = images.tensor.size()[2:]
        (height, width) = image_shape
        sheep = torch.FloatTensor([width, height, width, height]).to(self.device)
        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            if self.target == 'BOX':
                gt_elements = torch.stack([x.gt_boxes.tensor for x in gt_instances])
                gt_elements = gt_elements/sheep
            elif self.target == 'STATE':
                gt_elements = torch.stack([x.state for x in gt_instances])
            gt_mask = torch.stack([x.gt_box_id for x in gt_instances])

            losses = self.losses((gt_elements, gt_mask), (out_elements, mask_logits))
            return losses
        else:
            # TODO only support single image for now, need to support batch
            instances = Instances(image_shape)
            keep = mask_logits > self.score_thresh_test
            boxes = out_elements[keep]
            if self.target == 'BOX':
                boxes = boxes*sheep
                instances.pred_boxes = Boxes(boxes)
                instances.scores = mask_logits[keep]
                instances.pred_classes = torch.zeros_like(instances.scores)
            elif self.target == 'STATE':
                instances.state = boxes
                instances.scores = mask_logits[keep]

            result = {'instances': instances}

            return [result]

    def losses_with_bce(self, gt_set, predicted_set):
        matched_idx, set_loss = hungarian_loss(gt_set[0], predicted_set[0], gt_set[1], self.smooth_l1_beta)
        gt_id = [gt_set[1][i, matched_idx[i][1]] for i in range(gt_set[1].size(0))]
        gt_id = torch.stack(gt_id).float()
        return {
            "loss_cls": self.mask_loss(predicted_set[1], gt_id),
            "loss_set": 10.0*set_loss,
        }

    def losses_eu(self, gt_set, predicted_set):
        gt_data = torch.cat((gt_set[0], gt_set[1].unsqueeze(-1)), -1)
        pd_data = torch.cat((predicted_set[0], predicted_set[1].unsqueeze(-1)), -1)
        _, set_loss = hungarian_loss(gt_data, pd_data, gt_set[1], self.smooth_l1_beta)
        return {"loss_set": set_loss,}

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class ExchangeableDecoder(nn.Module):
    """
    set decoder with exchangeable structure
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        c_channels = cfg.MODEL.FLOWSETNET.CODE_CHANNELS
        embedding = cfg.MODEL.FLOWSETNET.EMBEDDING_TYPE
        element_dim    = cfg.MODEL.FLOWSETNET.ELEMENT_DIM
        max_num   = cfg.MODEL.FLOWSETNET.MAX_ELEMENT_NUM
        gnn_iter = cfg.MODEL.FLOWSETNET.GNN_ITERATION
        device = torch.device(cfg.MODEL.DEVICE)

        self.em_encoder = MLP(element_dim + 1, c_channels, [512], False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fi_encoder = MLP(input_shape[-1].channels, c_channels*2, [512, 512], False, (c_channels, 2))
        if embedding == 'PAIR_WISE':
            embedder_list = []
            for i in range(gnn_iter):
                embedder_list.append(PairwiseEmbedding(c_channels))
            self.embedder = nn.Sequential(*embedder_list)
        self.decoder = MLP(c_channels, element_dim + 1, [])
        self.init_ep = torch.nn.Parameter(torch.empty(1, max_num, element_dim+1).normal_().to(device))

    def forward(self, x):
        init_ep = self.init_ep.expand((x[-1].size(0), *self.init_ep.size()[1:]))

        x = self.avgpool(x[-1])
        x = torch.flatten(x, 1)
        fi_code = self.fi_encoder(x)
        em_code = self.em_encoder(init_ep)
        x = fi_code[...,0].unsqueeze(1)*em_code + fi_code[...,1].unsqueeze(1)

        if self.embedder:
            x = self.embedder(x)
        x = self.decoder(x)
        x_elements = x[..., :-1]
        x_mask = nn.Sigmoid()(x[..., -1])

        return x_elements, x_mask


class PairwiseEmbedding(nn.Module):
    def __init__(self, c_channels):
        super().__init__()

        self.pi_encoder = MLP(c_channels, c_channels*2, [512], False, (c_channels, 2))
        self.ni_encoder = MLP(c_channels, c_channels, [512], False)

    def forward(self, x):
        pi_code = self.pi_encoder(x)
        x = pi_code[..., 0].unsqueeze(-3) * x.unsqueeze(-2) + pi_code[..., 1].unsqueeze(-3)
        x = self.ni_encoder(x)
        x = x.mean(-2)
        return x


class MLPDecoder(nn.Module):
    """
    set decoder with MLP architecture
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        element_dim     = cfg.MODEL.FLOWSETNET.ELEMENT_DIM
        max_num         = cfg.MODEL.FLOWSETNET.MAX_ELEMENT_NUM
        c_channels      = cfg.MODEL.FLOWSETNET.CODE_CHANNELS

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = MLP(input_shape[-1].channels, (element_dim+1)*max_num, [512, c_channels], False, (max_num, element_dim+1))

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
             out_set(Tensor): transformed Tensor
        """
        x = self.avgpool(x[-1])
        x = torch.flatten(x, 1)
        x = self.decoder(x)
        x_elements = x[..., :-1]
        x_mask = nn.Sigmoid()(x[..., -1])

        return x_elements, x_mask


class MLP(nn.Module):
    """
    For Encoder and Decoders with an MLP architecture
    """
    def __init__(self, in_size, out_size, hidden_sizes, use_non_linear_output=False, out_shape=None, non_linear_layer=nn.ReLU):
        super().__init__()
        if out_shape:
            assert np.prod(np.array(out_shape)) == out_size
            self.out_shape = out_shape
        else:
            self.out_shape = (out_size, )
        layers = []
        in_sizes = [in_size] + hidden_sizes
        out_sizes = hidden_sizes + [out_size]
        sizes = list(zip(in_sizes, out_sizes))
        for (i, o) in sizes[0:-1]:
            layers.append(nn.Linear(i, o))
            layers.append(non_linear_layer())
        layers.append(nn.Linear(sizes[-1][0], sizes[-1][1]))
        if use_non_linear_output:
            layers.append(non_linear_layer())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        output_shape = x.size()[:-1] + self.out_shape
        return self.seq(x).view(output_shape)


@META_ARCH_REGISTRY.register()
def build_flowsetnet(cfg):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        flow_set_net (FlowSetNet): a FlowSetNet Module
    """

    decoder_type     = cfg.MODEL.FLOWSETNET.DECODER_TYPE
    in_features      = cfg.MODEL.FLOWSETNET.IN_FEATURES

    backbone = build_backbone(cfg)
    backbone_shape = backbone.output_shape()
    feature_shapes = [backbone_shape[f] for f in in_features]

    if decoder_type == 'MLP':
        decoder = MLPDecoder(cfg, feature_shapes)
    elif decoder_type == 'EXCHANGE':
        decoder = ExchangeableDecoder(cfg, feature_shapes)

    flow_set_net = FlowSetNet(cfg, backbone, decoder)
    return flow_set_net
