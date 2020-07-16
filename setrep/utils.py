import torch

from detectron2.structures import Boxes, BoxMode, Instances


def otter(axis, u, v):
    """ Compute outer product between u and v along axis (or u and u if v is not specified). """
    if v is None:
        v = u
    if axis == -1 or axis == len(u.size()):
        size_u = tuple(u.size())+ (v.size()[-1],)
        size_v = tuple(v.size()) + (u.size()[-1],)
    elif axis < len(u.size()):
        size_u = tuple(u.size()[:axis+1]) + (v.size()[axis],) + tuple(u.size()[axis+1:])
        size_v = tuple(v.size()[:axis]) + (u.size()[axis],) + tuple(v.size()[axis:])
    if axis>=0:
        axis = -len(u.size()) + axis
    u = u.unsqueeze(dim=axis).expand(*size_u)
    v = v.unsqueeze(dim=axis-1).expand(*size_v)
    return u, v

def annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
    """

    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)
    # target.state = torch.tensor([obj["state"] for obj in annos])

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    target.gt_box_id = torch.tensor([obj["box_identity"] for obj in annos])

    return target
