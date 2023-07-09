import click
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from utils import get_labelme_dataset_function
import os
import tqdm


@click.command()
@click.argument("directory", nargs=1)
def main(directory):
    class_labels = ['Branching 3D', 'Cups', 'Fan 2D', 'Hydrocorals', 'Other', 'Pale', 'True crabs', 'Two-dimensional lamellate', 'Yellow', 'biogenic + sediment', 'rubble', 'volcanic', 'volcanic + sediment']
    class_labels += ['Balls', 'Bottlebrush', 'Branching 3D', 'Colonial', 'Cups', 'Fan 2D', 'Feather stars', 'Hydrocorals', 'Other', 'Sea stars', 'Spider crabs',
 'Tube-like forms', 'Two-dimensional lamellate', 'coral rubble', 'lost+found']
    class_labels += ['Balls', 'Bottlebrush', 'Branching 3D', 'Colonial', 'Crustaceans', 'Fan 2D', 'Feather stars', 'Hydrocorals', 'Mushroom', 'Other', 'Other anemones', 'Pale', 'Sea cucumbers', 'Sea stars', 'Squat lobsters', 'Stalked', 'Three-dimensional branching', 'Tube-like forms', 'Two-dimensional lamellate', 'Unbranched', 'Yellow', 'biogenic rubble', 'lost+found', 'volcanic, sediment + biogenic rubble']
    
    class_labels = sorted(list(set(class_labels)))

    dataset_name = "ship_dataset"
    dataset_function = get_labelme_dataset_function(directory, class_labels)
    MetadataCatalog.get(dataset_name).thing_classes = class_labels
    DatasetCatalog.register(dataset_name, dataset_function)
    metadata = MetadataCatalog.get(dataset_name)
    for d in tqdm.tqdm(dataset_function()):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out = visualizer.draw_dataset_dict(d)


        points = d["annotations"]


        save_path = os.path.join(directory, "visualize", os.path.basename(d["file_name"]))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

        # cv2.imshow("Ground truth", out.get_image()[:, :, ::-1])
        # cv2.waitKey(1000)


if __name__ == "__main__":
    main()
