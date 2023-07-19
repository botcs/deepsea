import os
import cv2
import json
import numpy as np
import glob


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


def get_test_dataset_function(directory):
    def dataset_function():
        return test_directory_to_detectron_dataset(directory)
    return dataset_function

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