import os
import json
import contextlib
import io
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

DATA_PATH = '/scratch/chao/toy_datasets'
CLASSES = {
    "material": ["rubber", "metal"],
    "color": ["cyan", "blue", "yellow", "purple", "red", "green", "gray", "brown"],
    "shape": ["sphere", "cube", "cylinder"],
    "size": ["large", "small"],
}


def object_to_state(obj):
    coords = [p / 3 for p in obj["3d_coords"]]
    one_hot = lambda key: [int(obj[key] == x) for x in CLASSES[key]]
    material = one_hot("material")
    color = one_hot("color")
    shape = one_hot("shape")
    size = one_hot("size")
    assert sum(material) == 1
    assert sum(color) == 1
    assert sum(shape) == 1
    assert sum(size) == 1
    # concatenate all the classes
    return coords + material + color + shape + size

def extract_bounding_boxes(scene, split):
    """
    Code used for 'Object-based Reasoning in VQA' to generate bboxes
    https://arxiv.org/abs/1801.09718
    https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py#L51-L107
    """
    objs = scene["objects"]
    rotation = scene["directions"]["right"]

    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for i, obj in enumerate(objs):
        [x, y, z] = obj["pixel_coords"]

        [x1, y1, z1] = obj["3d_coords"]

        cos_theta, sin_theta, _ = rotation

        x1 = x1 * cos_theta + y1 * sin_theta
        y1 = x1 * -sin_theta + y1 * cos_theta

        height_d = 6.9 * z1 * (15 - y1) / 2.0
        height_u = height_d
        width_l = height_d
        width_r = height_d

        if obj["shape"] == "cylinder":
            d = 9.4 + y1
            h = 6.4
            s = z1

            height_u *= (s * (h / d + 1)) / ((s * (h / d + 1)) - (s * (h - s) / d))
            height_d = height_u * (h - s + d) / (h + s + d)

            width_l *= 11 / (10 + y1)
            width_r = width_l

        if obj["shape"] == "cube":
            height_u *= 1.3 * 10 / (10 + y1)
            height_d = height_u
            width_l = height_u
            width_r = height_u

        ymin.append((y - height_d))
        ymax.append((y + height_u))
        xmin.append((x - width_l))
        xmax.append((x + width_r))

    return [xmin, ymin, xmax, ymax]


def get_clevr_dicts(data_dir, split, max_objects):
    json_file = os.path.join(data_dir, "scenes", "CLEVR_"+split+"_scenes.json")
    with open(json_file) as f:
        scenes = json.load(f)['scenes']

    dataset_dicts = []
    for idx, v in enumerate(scenes):
        record = {}

        filename = os.path.join(data_dir, 'images', split, v["image_filename"])

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = 320.0
        record["width"] = 480.0

        objs = []
        boxes = extract_bounding_boxes(v, split)
        state = [object_to_state(obj) for obj in v["objects"]]

        num_objects = len(boxes[0])
        # pad with 0s for training dataset
        if split == 'train':
            if num_objects < max_objects:
                boxes = [dim + [0.0]*(max_objects - num_objects) for dim in boxes]
                state = state + [[0.0]*18]*(max_objects - num_objects)
            # fill in masks
            mask = [1.0]*num_objects + [0.0]*(max_objects-num_objects)
        else:
            mask = [1.0]*num_objects

        idx = list(range(len(boxes[0])))
        for i in idx:
            obj = {
                "state": state[i],
                "bbox": [boxes[0][i], boxes[1][i], boxes[2][i], boxes[3][i]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "box_identity": mask[i],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def get_tcoco_dicts(data_dir, split, max_objects):
    json_file = os.path.join(data_dir, "annotations", "instances_"+split+"2017.json")

    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))

    dataset_dicts = []
    for (img_dict, anno_dict_list) in imgs_anns:
        for anno in anno_dict_list:
            if anno['iscrowd'] == 1:
                anno_dict_list.remove(anno)

        record = {}
        record["file_name"] = os.path.join(data_dir, split+'2017', img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        num_objects = 0.0
        if len(anno_dict_list) == 0 or len(anno_dict_list) > max_objects:
            continue
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            bbox = anno['bbox']
            bbox = [bbox[0], bbox[1],(bbox[0] + bbox[2]), (bbox[1] + bbox[3])]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,  # the REL is not supported, but I am using it as REL
                "box_identity": 1.0,
                "category_id": 0,
                "iscrowd": anno['iscrowd']
            }
            objs.append(obj)
            num_objects = num_objects + 1
        if split == 'train':
            while num_objects < max_objects:
                bbox = [0.0,0.0,0.0,0.0]
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,  # the REL is not supported, but I am using it as REL
                    "box_identity": 0.0,
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
                num_objects = num_objects + 1
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_data(dataset_name):
    if dataset_name == 'clevr':
        for d in ["train", "val", "test"]:
            path = os.path.join(DATA_PATH, "clevr")
            DatasetCatalog.register("clevr_" + d, lambda d=d: get_clevr_dicts(path, d, 10))
            MetadataCatalog.get("clevr_" + d).set(thing_classes=["box"])
    if dataset_name == 'tcoco':
        for d in ["train", "val", "test"]:
            path = os.path.join(DATA_PATH, "coco")
            DatasetCatalog.register("tcoco_" + d, lambda d=d: get_tcoco_dicts(path, d, 20))
            MetadataCatalog.get("tcoco_" + d).set(thing_classes=["box"])
