import click
import numpy as np
import os
import torch
from utils import get_labelme_dataset_function

from detectron2 import model_zoo
from detectron2.data import detection_utils as utils, transforms as T, build_detection_train_loader
from detectron2.engine import launch, DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.evaluation import RotatedCOCOEvaluator, DatasetEvaluators
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog

import cv2


def convert_rotated_bbox_to_polygon(bbox):
    # Convert the rotated bounding box to 
    # x1 y1 x2 y2 x3 y3 x4 y4 format
    cx, cy, w, h, angle = bbox
    ret = cv2.boxPoints(((cx, cy), (w, h), angle))
    return ret

    
    
def convert_polygon_to_rotated_bbox(polygon):
    # Convert the polygon to rotated bounding box
    points = np.array(polygon, dtype=np.float32)
    ((cx, cy), (w, h), a) = cv2.minAreaRect(points)

    if w < h:
        h_temp = h
        h = w
        w = h_temp
        a += 90

    a = (360 - a) % 360  # ccw [0, 360]

    # Clamp to [0, 90] and [270, 360]
    if (a > 90) and (a <= 180):
        a -= 180
    elif (a > 180) and (a < 270):
        a -= 180

    # Clamp to [-180, 180]
    if a > 180:
        a -= 360

    return [cx, cy, w, h, a]
    
def rotate_bbox(annotation, transforms):
    # converted_bbox = convert_rotated_bbox_to_polygon(annotation['bbox'])
    bbox8 = annotation['bbox8']
    bbox8 = np.asarray(bbox8).reshape(4, 2)
    converted_bbox = transforms.apply_coords(bbox8)
    if len(converted_bbox) == 0:
        # After augmentation, some boxes are degenerate and become empty.
        # this hack makes sure that this box is ignored.
        annotation['iscrowd'] = 1
        annotation['bbox'] = [0, 0, 0, 0, 0]
    else:
        annotation['bbox'] = convert_polygon_to_rotated_bbox(converted_bbox)
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    # annotation["bbox"] = transforms.apply_rotated_box(
    #     np.asarray([annotation['bbox']]))[0]
    # annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation


def get_shape_augmentations():
    # Optional shape augmentations
    return [
        T.RandomFlip(),
        T.RandomRotation(angle=[-10, 10], sample_style="range", expand=True),
        # T.RandomCrop("relative_range", (0.85, 0.85)),
        T.MinIoURandomCrop(min_ious=(0.7, 0.8, 0.9), min_crop_size=0.5),
        T.ResizeShortestEdge(short_edge_length=(
            640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'),
    ]


def get_color_augmentations():
    # Optional color augmentations
    return T.AugmentationList([
        T.RandomBrightness(0.9, 1.1),
        T.RandomSaturation(intensity_min=0.75, intensity_max=1.25),
        T.RandomContrast(intensity_min=0.76, intensity_max=1.25),
        T.RandomLighting(0.7),
    ])



def dataset_mapper(dataset_dict):

    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    color_aug_input = T.AugInput(image)
    get_color_augmentations()(color_aug_input)
    image = color_aug_input.image
    # # draw rotated rectangles with cv2 before augmentation
    # before_image = image.copy()
    # for annotation in dataset_dict["annotations"]:
    #     bbox = annotation['bbox']
    #     cx, cy, w, h, angle = bbox
    #     ret = cv2.boxPoints(((cx, cy), (w, h), -angle))
    #     ret = np.intp(ret)
    #     before_image = cv2.drawContours(before_image, [ret], 0, (0, 0, 255), 2)
    # cv2.imwrite('before.jpg', before_image)

    image, image_transforms = T.apply_transform_gens(
        get_shape_augmentations(), image)
    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))

    annotations = [
        rotate_bbox(obj, image_transforms)
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances_rotated(
        annotations, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    # # draw rotated rectangles with cv2
    # after_image = image.copy()
    # for annotation in annotations:
    #     bbox = annotation['bbox']
    #     cx, cy, w, h, angle = bbox
    #     ret = cv2.boxPoints(((cx, cy), (w, h), -angle))
    #     ret = np.intp(ret)
    #     after_image = cv2.drawContours(after_image, [ret], 0, (0, 0, 255), 2)
    # cv2.imwrite('after.jpg', after_image)
    # import ipdb; ipdb.set_trace()

    return dataset_dict


class RotatedBoundingBoxTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [RotatedCOCOEvaluator(
            dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=dataset_mapper)


def train_detectron(flags):
    class_labels = ['Balls', 'Bottlebrush', 'Branching 3D', 'Colonial', 'Crustaceans', 'Cups', 'Fan 2D', 'Feather stars', 'Hydrocorals', 'Mushroom', 'Other', 'Other anemones', 'Pale', 'Prawns / Shrimps / Mysids', 'Sea cucumbers', 'Sea stars', 'Spider crabs', 'Squat lobsters', 'Stalked', 'Three-dimensional branching', 'True crabs', 'Tube-like forms', 'Two-dimensional lamellate', 'Unbranched', 'Yellow', 'biogenic + sediment', 'biogenic rubble', 'coral rubble', 'lost+found', 'rubble', 'volcanic', 'volcanic + sediment', 'volcanic, sediment + biogenic rubble']

    class_labels = sorted(list(set(class_labels)))

    dataset_function = get_labelme_dataset_function(
        flags["directory"], class_labels)
    dataset_name = "dive8file4"
    MetadataCatalog.get(dataset_name).thing_classes = class_labels
    DatasetCatalog.register(dataset_name, dataset_function)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))  # Base model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Weights
    
    # cfg.merge_from_file(model_zoo.get_config_file(
    #     "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Base model
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Weights
    
    # Rotated bbox specific config in the same directory as this file
    cfg.merge_from_file(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "rotated_bbox_config.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)
    # Directory where the checkpoints are saved, "." is the current working dir
    cfg.OUTPUT_DIR = "training-output/augmentation-tests/2/dive13-training/"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)
    cfg.DATALOADER.NUM_WORKERS = 8

    # save the config to a file for reference
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg_filename = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(cfg_filename, 'w') as f:
        f.write(cfg.dump())


    trainer = RotatedBoundingBoxTrainer(cfg)
    # NOTE: important, the model will not train without this
    trainer.resume_or_load(resume=False)
    trainer.checkpointer.max_to_keep = 10
    trainer.train()


@click.command()
@click.argument('directory', nargs=1)
@click.option('--num-gpus', default=0, help='Number of GPUs to use, default none')
def main(**flags):
    launch(
        train_detectron,
        flags["num_gpus"],
        args=(flags,),
    )


if __name__ == "__main__":
    main()