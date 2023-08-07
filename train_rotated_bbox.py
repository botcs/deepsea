import argparse
import numpy as np
import json
import os
import torch

from detectron2 import model_zoo
from detectron2.data import detection_utils as utils, transforms as T, build_detection_train_loader
from detectron2.engine import launch, DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.evaluation import RotatedCOCOEvaluator, DatasetEvaluators
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from utils import check_overlap

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
    converted_bbox = transforms.apply_coords(bbox8)
    if len(converted_bbox) == 0:
        # After augmentation, some boxes are degenerate and become empty.
        # this hack makes sure that this box is ignored.
        annotation['iscrowd'] = 1
        annotation['bbox'] = [0, 0, 0, 0, 0]
    else:
        annotation['bbox'] = convert_polygon_to_rotated_bbox(converted_bbox)
    annotation['bbox8'] = converted_bbox
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation


def get_shape_augmentations():
    # Optional shape augmentations
    return [
        T.RandomFlip(),
        T.RandomRotation(angle=[-15, 15], sample_style="range", expand=True),
        # T.RandomCrop("relative_range", (0.85, 0.85)),
        # T.MinIoURandomCrop(min_ious=(1.0,), min_crop_size=0.5, ),
        # T.ResizeShortestEdge(short_edge_length=(
        #     250, 300, 350, 400, 450), max_size=1333, sample_style='choice'),
    ]


def get_color_augmentations():
    # Optional color augmentations
    return T.AugmentationList([
        T.RandomBrightness(0.9, 1.1),
        T.RandomSaturation(intensity_min=0.75, intensity_max=1.25),
        T.RandomContrast(intensity_min=0.76, intensity_max=1.25),
        T.RandomLighting(0.7),
    ])


def crop_around_bbox(image, annotations, margin=0.4, crop_or_pad='pad'):
    # crop the image that might contain false positives
    # by using the bounding box of the instances

    # if there are no annotations, return the original image
    if len(annotations) == 0:
        return image, annotations, [0, 0, image.shape[1], image.shape[0]]
    
    # if there is only one annotation, change the margin to .8
    # so that the crop is not too tight
    if len(annotations) == 1:
        margin = 0.8

    all_bboxes = [annotation['bbox8'] for annotation in annotations]
    all_bboxes = np.concatenate(all_bboxes, axis=0)
    all_bboxes = np.intp(all_bboxes)

    # find the minimum bounding box that contains all the bounding boxes
    # format of the bounding box is x, y, w, h
    min_bbox = cv2.boundingRect(all_bboxes)
    
    # add a small margin to the crop
    # format of the crop is x1, y1, x2, y2
    crop_bbox = [
        int(min_bbox[0] - margin * min_bbox[2]),
        int(min_bbox[1] - margin * min_bbox[3]),
        int(min_bbox[0] + (1 + margin) * min_bbox[2]),
        int(min_bbox[1] + (1 + margin) * min_bbox[3]),
    ]
    
    # make sure the crop is within the image
    crop_bbox[0] = max(0, crop_bbox[0])
    crop_bbox[1] = max(0, crop_bbox[1])
    crop_bbox[2] = min(image.shape[1], crop_bbox[2])
    crop_bbox[3] = min(image.shape[0], crop_bbox[3])

    # crop the image
    height, width, _ = image.shape
    image = image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]

    if crop_or_pad == 'pad':
        new_image = np.zeros((height, width, 3), dtype=np.uint8)
        new_image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :] = image
        image = new_image

    if crop_or_pad == 'crop':
        for annotation in annotations:
            # adjust the bounding boxes to the crop 
            # (only the cx and cy need to be adjusted)
            annotation['bbox'][0] -= crop_bbox[0]
            annotation['bbox'][1] -= crop_bbox[1]

            # adjust the bounding box8 to the crop
            annotation['bbox8'][:, 0] -= crop_bbox[0]
            annotation['bbox8'][:, 1] -= crop_bbox[1]
        

    return image, annotations, crop_bbox


def debug_write_image(dataset_dict, filename):
    image = np.asarray(dataset_dict["image"].copy())
    for annotation in dataset_dict["annotations"]:
        bbox = annotation['bbox']
        cx, cy, w, h, angle = bbox
        cx += 1
        ret = cv2.boxPoints(((cx, cy), (w, h), -angle))
        ret = np.intp(ret)

        bbox8 = annotation['bbox8']
        bbox8 = np.intp(bbox8)
        image = cv2.drawContours(image, [ret], 0, (255, 255, 255, 128), 2)
        image = cv2.drawContours(image, [bbox8], 0, (255, 0, 0, 128), 2)
    cv2.imwrite(filename, image)



class DatasetMapper:
    def __init__(self, cfg, num_paste=0):
        self.dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])

        self.color_aug = get_color_augmentations()
        self.shape_aug = get_shape_augmentations()
        self.num_paste = num_paste

    def __call__(self, dataset_dict):
        main_dict = self.map_single(dataset_dict, crop_or_pad='pad')
        main_dict["pasted_crop_bboxes8"] = [main_dict["self_crop_bbox8"]]

        for i in range(self.num_paste):
            # sample a random image
            paste_dict = np.random.choice(self.dataset_dicts)
            paste_dict = self.map_single(paste_dict, crop_or_pad='crop')

            # apply augmentations
            self.augment(paste_dict)

            main_dict = self.paste(main_dict, paste_dict)

        image = main_dict["image"]
        annotations = main_dict["annotations"]

        # map image to torch.Tensor
        image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        main_dict["image"] = image

        # map annotations to Instances
        instances = utils.annotations_to_instances_rotated(
            annotations, image.shape[1:3]
        )
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


    def paste(self, main_dict, paste_dict):
        """
        Paste the image and annotations from paste_dict to main_dict randomly
        """

        base_image = main_dict["image"].copy()
        base_annotations = main_dict["annotations"]
        base_crop_bboxes = main_dict["pasted_crop_bboxes8"]

        paste_image = paste_dict["image"]
        paste_annotations = paste_dict["annotations"]
        paste_crop_bbox = paste_dict["self_crop_bbox8"]

        if base_image.shape[0] <= paste_image.shape[0] or base_image.shape[1] <= paste_image.shape[1]:
            # if the paste image is larger than the base image
            # just return the base image and annotations
            return main_dict
        
        # sample a random location to paste the image
        # until there is no overlap with the base image
        N = 1000
        for trial in range(N):
            paste_xmin = np.random.randint(0, base_image.shape[0] - paste_image.shape[0])
            paste_ymin = np.random.randint(0, base_image.shape[1] - paste_image.shape[1])
            
            paste_xmax = paste_xmin + paste_image.shape[0]
            paste_ymax = paste_ymin + paste_image.shape[1]
            
            candidate_paste_crop_bbox8 = paste_crop_bbox.copy()
            candidate_paste_crop_bbox8[:, 0] += paste_ymin
            candidate_paste_crop_bbox8[:, 1] += paste_xmin

            # check if there is any overlap
            overlap = check_overlap(
                base_crop_bboxes, candidate_paste_crop_bbox8
            )
            if not overlap:
                base_crop_bboxes.append(candidate_paste_crop_bbox8)
                break

        if trial == N - 1:
            # if there is no location that does not overlap
            # just return the base image and annotations
            return main_dict

        # paste the image
        base_image[paste_xmin:paste_xmax, paste_ymin:paste_ymax] = paste_image

        # adjust the bounding boxes to the paste location
        for annotation in paste_annotations:
            annotation['bbox'][0] += paste_ymin
            annotation['bbox'][1] += paste_xmin

            annotation['bbox8'][:, 0] += paste_ymin
            annotation['bbox8'][:, 1] += paste_xmin

        # add the annotations to the base annotations
        base_annotations.extend(paste_annotations)

        main_dict["image"] = base_image
        main_dict["annotations"] = base_annotations
        main_dict["pasted_crop_bboxes8"] = base_crop_bboxes

        return main_dict



    def augment(self, dataset_dict):
        image = dataset_dict["image"]

        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("Augmentation error: image shape is invalid")
        
        if image.dtype != np.uint8:
            raise ValueError("Augmentation error: image dtype is not uint8")

        annotations = dataset_dict["annotations"]
        self_crop_bbox = dataset_dict["self_crop_bbox8"]

        color_aug_input = T.AugInput(image)
        self.color_aug(color_aug_input)
        image = color_aug_input.image

        image, image_transforms = T.apply_transform_gens(
            self.shape_aug, 
            image
        )
        
        annotations = [
            rotate_bbox(obj, image_transforms)
            for obj in annotations
            if obj.get("iscrowd", 0) == 0
        ]
        
        self_crop_bbox = image_transforms.apply_coords(self_crop_bbox)

        dataset_dict["image"] = image
        dataset_dict["annotations"] = annotations
        dataset_dict["self_crop_bbox8"] = self_crop_bbox

    def map_single(self, dataset_dict, crop_or_pad):
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        dataset_dict["image"] = image
        annotations = dataset_dict["annotations"]

        # convert all bbox8 to np.array[4, 2]
        for annotation in annotations:
            annotation['bbox8'] = np.asarray(annotation['bbox8']).reshape(4, 2)


        # crop the image that might contain false positives
        # by using the bounding box of the instances
        image, annotations, crop_bbox = crop_around_bbox(
            image, annotations, margin=0.2, crop_or_pad=crop_or_pad
        )

        dataset_dict["image"] = image
        dataset_dict["annotations"] = annotations

        # for rotation augmentations, the reference crop_bbox needs to be adjusted
        if crop_or_pad == 'crop':
            crop_bbox8 = np.array([
                [0, 0],
                [image.shape[1], 0],
                [image.shape[1], image.shape[0]],
                [0, image.shape[0]],
            ])
        else:
            crop_bbox8 = np.array([
                [crop_bbox[0], crop_bbox[1]],
                [crop_bbox[2], crop_bbox[1]],
                [crop_bbox[2], crop_bbox[3]],
                [crop_bbox[0], crop_bbox[3]],
            ])
        dataset_dict["self_crop_bbox8"] = crop_bbox8
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
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, num_paste=3))


def train_detectron(args):
    class_labels = open(args.class_labels).read().splitlines()
    train_dataset_function = lambda: json.load(open(args.train_json))
    val_dataset_function = lambda: json.load(open(args.val_json))


    # Register the datasets
    train_dataset_name = "0807_train"
    val_dataset_name = "0807_val"
    MetadataCatalog.get(train_dataset_name).set(thing_classes=class_labels)
    MetadataCatalog.get(val_dataset_name).set(thing_classes=class_labels)
    DatasetCatalog.register(train_dataset_name, train_dataset_function)
    DatasetCatalog.register(val_dataset_name, val_dataset_function)

    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file(
    #     "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))  # Base model
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Weights
    
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Base model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Weights
    
    # Rotated bbox specific config in the same directory as this file
    cfg.merge_from_file(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "rotated_bbox_config.yaml"))
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    # Directory where the checkpoints are saved, "." is the current working dir
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.TEST.EVAL_PERIOD = 200
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    

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





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--train-json", help="Path to the training json file", required=True)
    parser.add_argument("--val-json", help="Path to the validation json file")
    parser.add_argument("--class-labels", help="Path to the class labels file", required=True)
    parser.add_argument("--output-dir", help="Path to the directory for the experiment output", default="./debug0806")
    args = parser.parse_args()

    launch(
        train_detectron,
        num_gpus_per_machine=args.num_gpus,
        args=(args,),
    )

