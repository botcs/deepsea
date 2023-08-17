import os
import cv2
import json
import numpy as np
import glob
import tabulate
import pandas as pd
from shapely.geometry import Polygon

def check_overlap(set_of_bboxes8, test_bbox8):
    set_of_bboxes8 = [Polygon(bbox8) for bbox8 in set_of_bboxes8]
    test_bbox8 = Polygon(test_bbox8)
    for bbox8 in set_of_bboxes8:
        if bbox8.intersects(test_bbox8):
            return True
        
    return False
    


def convert_4xy_to_cxcywha(points):
    if isinstance(points, list):
        points = np.array(points, dtype=np.float32)

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

    return (cx, cy, w, h, a)


def get_labelme_dataset_function(labelme_directory, class_labels):
    def dataset_function():
        return labelme_directory_to_detectron_dataset(labelme_directory, class_labels)
    return dataset_function


def labelme_directory_to_detectron_dataset(directory, class_labels):
    files = glob.glob(directory + '/**/*.json', recursive=True)
    images = []
    classes = []
    counter = 0
    for filename in files:
        if '.json' not in filename:
            continue
        path = os.path.join(directory, filename)
        file_base = path.split('.json')[0]
        color_jpg = f"{file_base}.jpg"
        color_png = f"{file_base}.png"

        if os.path.exists(color_jpg):
            suffix = "jpg"
        elif os.path.exists(color_png):
            suffix = "png"
        else:
            continue
        with open(path, 'rt') as f:
            data = json.load(f)

        annotations = []

        for shape in data['shapes']:
            if shape['label'] not in classes:
                classes.append(shape['label'])
            points = np.array(shape['points'], dtype=np.float32)
            cx, cy, w, h, a = convert_4xy_to_cxcywha(points)

            annotations.append({
                "bbox_mode": 4,  # Oriented bounding box (cx, cy, w, h, a)
                "category_id": class_labels.index(shape['label']),
                "bbox": (cx, cy, w, h, a),
                "bbox8": points.flatten().tolist(),
            })

        uid = os.path.basename(filename).split('.')[0]
        images.append({
            "id": uid,
            "image_id": counter,
            "file_name": f"{file_base}.{suffix}",
            "height": data['imageHeight'],
            "width": data['imageWidth'],
            "annotations": annotations
        })
        counter += 1
    
    return images


def print_class_counts(df):
    classes = df.label_name.unique().tolist()
    classes.sort(key=lambda x: df[df.label_name == x].shape[0], reverse=True)
    print(tabulate.tabulate(
        [(i, c, df[df.label_name == c].shape[0]) for i, c in enumerate(classes)],
        headers=["index", "category", "count"]
    ))
    
    print("Total number of annotations: ", df.shape[0])
    print("Total number of images: ", len(df.frame_id.unique()))


def biigle_dataframe_to_detectron_dataset(data, image_dir, class_labels):
    # detectron2 requires a list of dictionaries, one dictionary per image
    detectron_data = {}

    for i, row in data.iterrows():
        video_id = row.video_id
        frame = row.frame
        filename = f"{video_id}_{frame}.jpg"
        
        annotations = []
        per_row_points = json.loads(row.points)
        
        for points in per_row_points:
            points = [[p1, p2] for p1, p2 in zip(points[::2], points[1::2])]
            points = np.array(points, dtype=np.float32)
            class_label = row.label_name
            cx, cy, w, h, a = convert_4xy_to_cxcywha(points)
        
            annotations.append({
                "bbox_mode": 4,  # Oriented bounding box (cx, cy, w, h, a)
                "category_id": class_labels.index(class_label),
                "bbox": (cx, cy, w, h, a),
                "bbox8": points.flatten().tolist(),
            })

        uid = f"{video_id}_{frame}"
        if uid not in detectron_data:
            detectron_data[uid] = {
                "image_id": uid,
                "file_name": os.path.join(image_dir, str(video_id), filename),
                "height": 1080,
                "width": 1920,
                "video_id": video_id,
                "timestamp": frame,
                "annotations": annotations,
            }
        else:
            detectron_data[uid]["annotations"].extend(annotations)

    detectron_data = list(detectron_data.values())
    return detectron_data

def get_test_dataset_function(directory):
    def dataset_function():
        return test_directory_to_detectron_dataset(directory)
    return dataset_function


def csvdir_to_single_df(biigle_csv_dir):
    """
    Joins the biigle csv files in the given directory and returns a pandas dataframe.

    """
    csv_files = glob.glob(os.path.join(biigle_csv_dir, '*.csv'))
    all_df = pd.DataFrame()
    for csv_file in csv_files:
        video_df = pd.read_csv(csv_file)
        all_df = pd.concat([all_df, video_df])

    return all_df


def test_directory_to_detectron_dataset(directory):
    files = glob.glob(os.path.join(directory, '*.jpg'))
    files.sort()
    images = []
    counter = 0
    for filename in files:
        if '.jpg' not in filename:
            raise RuntimeError(f"Expected .jpg file, got {filename}")
        
        width, height = cv2.imread(filename).shape[:2]
        image_basename = os.path.splitext(os.path.basename(filename))[0]
        images.append({
            "id": image_basename,
            "image_id": counter,
            "file_name": filename,
            "width": width,
            "height": height,
        })
        counter += 1
    
    return images

def collect_class_labels(biigle_csv_dir, class_labels_file_path):
    """
    Collects the class labels from the biigle csv files in the given directory
    and writes them to the given file path.

    """
    csv_files = glob.glob(os.path.join(biigle_csv_dir, '*.csv'))
    class_labels = set()
    all_df = pd.DataFrame()
    for csv_file in csv_files:
        video_df = pd.read_csv(csv_file)
        class_labels.update(video_df.label_name.unique().tolist())
        all_df = pd.concat([all_df, video_df])

    class_labels = list(class_labels)
    # sort the class labels based on frequency
    class_labels.sort(key=lambda x: all_df[all_df.label_name == x].shape[0], reverse=True)

    
    # print a table of categories, indices and their counts
    print(tabulate.tabulate(
        [(i, c, all_df[all_df.label_name == c].shape[0]) for i, c in enumerate(class_labels)],
        headers=["index", "category", "count"]
    ))
    


    # save the class labels to a file
    with open(class_labels_file_path, 'w') as f:
        f.write('\n'.join(class_labels))


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--biigle-csv", help="Path to the biigle csv file", required=True)
    # parser.add_argument("--image-dir", help="Path to the directory containing the images", required=True)
    # parser.add_argument("--output-dir", help="Path to the directory to save the output json file", default="./")
    # parser.add_argument("--class-labels", help="Path to the class labels file", default="./class_labels.txt")

    # args = parser.parse_args()

    # print(f"\n\n\n\nProcessing biigle csv file... {args.biigle_csv}")
    
    # class_labels = open(args.class_labels, 'r').read().splitlines()
    # dataset = biigle_csv_to_detectron_dataset(args.biigle_csv, args.image_dir, class_labels)
    pass