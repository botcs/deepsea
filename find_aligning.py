import click
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from utils import get_labelme_dataset_function
import os
import tqdm
import json
import numpy as np
import glob


def convert_jpgs_to_video(jpg_folder, output_video_path, fps):
    try:
        # Get the list of JPG files in the folder
        jpg_files = [f for f in os.listdir(jpg_folder) if f.endswith('.jpg')]
        
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
    except Exception as e:
        print(f"Error creating video: {e}")


def get_extension_and_filename(path):
    filename = os.path.basename(path)
    filename, ext = os.path.splitext(filename)
    return ext, filename

@click.command()
@click.argument("directory", nargs=1)
def main(directory):
    annot_file = os.path.join(directory, "902.0358922953482.json")
    # get the annotations
    annot = json.load(open(annot_file, "rt"))

    shapes = annot["shapes"]

    os.makedirs(os.path.join(directory, "visualize"), exist_ok=True)
    file_list = glob.glob(os.path.join(directory, "*.jpg"))
    # filename is like: 908.4399999999878.jpg
    
    file_list.sort(key=lambda x: float(os.path.splitext(os.path.basename(x))[0]))
    
    for i, img_path in enumerate(tqdm.tqdm(file_list)):
        if ".jpg" not in img_path:  
            continue
        img = cv2.imread(os.path.join(directory, img_path))
        if img is None:
            print(f"Could not read {img_path}")
        for shape in shapes:
            points = np.array(shape['points'], dtype=np.int32)
            # points = [[p1, p2] for p1, p2 in zip(points[::2], points[1::2])]
            # points = np.array(points, dtype=np.int32)
            img = cv2.polylines(img, [points], True, (0, 255, 0), 2)


        save_path = os.path.join(directory, "visualize", f"{i:05d}.jpg")
        cv2.imwrite(save_path, img[:, :, ::-1])

    # turn the images into a video
    convert_jpgs_to_video(os.path.join(directory, "visualize"), os.path.join(directory, "video.mp4"), 5)    




if __name__ == "__main__":
    main()
