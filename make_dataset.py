#!/usr/bin/env python

"""
This script is used to make the dataset for training and testing.

The only requirement is a Biigle csv file of the annotations.

e.g. dive8-file4.csv is the input file

The script uses the following sub-scripts:
python convert_mp4_to_images.py --video file4dive8jc66_1.mp4 --biigle ~/Downloads/11715-dive8-file4.csv --out ~/storage/deepsea/dataset/dive8-file4
python biigle2labelme.py ~/Downloads/11715-dive8-file4.csv --out ~/storage/deepsea/dataset/dive8-file4/
"""

import os
import pandas as pd
from convert_mp4_to_images import extract_frames_from_video, save_frames_to_disk
from biigle2labelme import convert_biigle_to_labelme
import json
import boto3

def download_mp4_if_not_exists(csv_path, out_dir):

    def download_mp4_from_s3(bucket_name, object_key, destination_path):
        s3 = boto3.client('s3')
        try:
            s3.download_file(bucket_name, object_key, destination_path)
            print("File downloaded successfully.")
        except Exception as e:
            print(f"Error downloading file: {e}")

    df = pd.read_csv(csv_path)

    # assert data belongs to a single video
    assert df.video_filename.nunique() == 1
    video_filename = df.video_filename.unique()[0]

    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, video_filename)
    print(f"Checking if video {video_path} exists...")
    # check if video exists
    if not os.path.exists(video_path):
        # download video
        print("Downloading video...")
        download_mp4_from_s3("biigle-deepsea", video_filename, video_path)
    else:
        print("Video already exists.")

    return video_path

def main(csv_path, out_dir):
    video_path = download_mp4_if_not_exists(csv_path, out_dir)
    # file4dive8jc66_1.mp4
    #     ^    ^
    file_number = int(os.path.basename(video_path).split("dive")[0].split("file")[1])
    dive_number = int(os.path.basename(video_path).split("dive")[1].split("jc")[0])

    label_timestamps = pd.read_csv(csv_path).frames.tolist()
    label_timestamps = [json.loads(ts) for ts in label_timestamps]
    # flatten the list
    label_timestamps = [item for sublist in label_timestamps for item in sublist]

    print(f"Extracting frames from video {video_path}...")
    # extract frames from video
    frames = extract_frames_from_video(video_path, label_timestamps)

    frames_dir = os.path.join(out_dir, "frames", "labeled", f"dive{dive_number}-file{file_number}")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Saving frames to {frames_dir}...")
    # save frames to disk
    save_frames_to_disk(frames, frames_dir)

    # convert biigle to labelme
    convert_biigle_to_labelme(csv_path, frames_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-paths", help="Path to the csv file", nargs="+", required=True)
    parser.add_argument("--output-dir", help="Output directory", required=True)
    args = parser.parse_args()

    for csv_path in args.csv_paths:
        main(os.path.expanduser(csv_path), os.path.expanduser(args.output_dir))
