import click
import cv2
import os
import tqdm

from utils import get_test_dataset_function

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def save_labelme_json(image_path, output_path, predictions):
    # Get the image dimensions
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Create the JSON dictionary
    json_dict = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }
    
    # Add the shapes
    for prediction in predictions:
        class_name = prediction["class_name"]
        bbox = prediction["bbox"]
        xmin, ymin, xmax, ymax, angle = bbox
        
        shape_dict = {
            "label": class_name,
            "points": [
                [xmin, ymin],
                [xmax, ymax]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        json_dict["shapes"].append(shape_dict)
    
    # Save the JSON file
    json_path = os.path.join(output_path, os.path.basename(image_path).replace(".jpg", ".json"))
    with open(json_path, "wt") as f:
        json.dump(json_dict, f, indent=2)
    
    print(f"Saved {json_path}")


def convert_jpgs_to_video(jpg_folder, output_video_path, fps):
    # Get the list of JPG files in the folder
    jpg_files = [f for f in os.listdir(jpg_folder) if (f.endswith('.jpg') or f.endswith('.png'))]
    
    # Sort the files in ascending order
    jpg_files.sort()
    
    # Get the first image dimensions
    first_image = cv2.imread(os.path.join(jpg_folder, jpg_files[0]))
    height, width, _ = first_image.shape
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec (e.g., 'XVID', 'MJPG')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Write each image to the video writer
    for jpg_file in jpg_files:
        image_path = os.path.join(jpg_folder, jpg_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # Release the video writer
    video_writer.release()
    
    print("Video created successfully.")


@click.command()
@click.argument('directory', nargs=1)
@click.option('--weights', default="model_final.pth", help='Path to the model to use')
def main(directory, weights):
    class_labels = [
        'Branching 3D', 
        'Cups', 
        'Fan 2D', 
        'Hydrocorals', 
        'Other', 
        'Pale', 
        'True crabs', 
        'Two-dimensional lamellate', 
        'Yellow', 
        'biogenic + sediment', 
        'rubble', 
        'volcanic', 
        'volcanic + sediment']
    color_palette = {
        'Branching 3D': (0, 85, 170),  # Deep Blue
        'Cups': (190, 0, 95),  # Magenta
        'Fan 2D': (0, 153, 115),  # Teal
        'Hydrocorals': (0, 170, 255),  # Sky Blue
        'Other': (255, 128, 0),  # Orange
        'Pale': (255, 204, 0),  # Gold
        'True crabs': (204, 0, 102),  # Raspberry
        'Two-dimensional lamellate': (102, 204, 0),  # Lime Green
        'Yellow': (255, 255, 0),  # Yellow
        'biogenic + sediment': (255, 77, 148),  # Coral
        'rubble': (153, 153, 153),  # Silver
        'volcanic': (204, 51, 0),  # Burnt Orange
        'volcanic + sediment': (170, 85, 0)  # Sienna
    }
    dataset_name = "dive8file4"

    dataset_function = get_test_dataset_function(directory)
    MetadataCatalog.get(dataset_name).thing_classes = class_labels
    MetadataCatalog.get(dataset_name).thing_colors = {
        i: color_palette[k] for i, k in enumerate(class_labels)
    }

    DatasetCatalog.register(dataset_name, dataset_function)
    metadata = MetadataCatalog.get(dataset_name)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Base model
    # Rotated bbox specific config in the same directory as this file
    cfg.merge_from_file(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "rotated_bbox_config.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35  # Preiction confidence threshold,

    predictor = DefaultPredictor(cfg)
    output_dir = os.path.join(directory, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    for i, d in enumerate(tqdm.tqdm(dataset_function())):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        # Save predictions in labelme format
        



        v = Visualizer(img[:, :, ::-1], metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow("Predictions", out.get_image()[:, :, ::-1])

        cv2.imwrite(f"{output_dir}/{d['id']}.png", out.get_image()[:, :, ::-1])
        # cv2.waitKey(1000)

    root_directory_name = os.path.basename(os.path.normpath(directory))
    convert_jpgs_to_video(output_dir, os.path.join(output_dir, f"{root_directory_name}.mp4"), 5)

if __name__ == "__main__":
    main()