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
import tabulate
import json
import boto3
import glob
from utils import biigle_dataframe_to_detectron_dataset, print_class_counts
import random

def download_mp4_if_not_exists(all_biigle_df, out_dir):

    def download_mp4_from_s3(bucket_name, object_key, destination_path):
        s3 = boto3.client('s3')
        
        # check if file exists in s3
        try:
            s3.head_object(Bucket=bucket_name, Key=object_key)
            print("File found on S3!")
        except:
            print("File not found on S3")
            raise RuntimeError(f"File {object_key} not found on S3")

        s3.download_file(bucket_name, object_key, destination_path)
        print("File downloaded successfully.")

    # # assert data belongs to a single video
    # assert df.video_filename.nunique() == 1
    # video_filename = df.video_filename.unique()[0]

    for video_filename in all_biigle_df.video_filename.unique():
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, video_filename)
        print(f"Checking if video {video_path} exists...")
        # check if video exists
        if not os.path.exists(video_path):
            # download video
            print("Downloading video...")
            download_mp4_from_s3("celiaproject", video_filename, video_path)
        else:
            print("Video already exists.")


def remove_unavailable_videos(all_biigle_df):
    """
    Check if the video exists on AWS S3.
    If not, remove the video and the corresponding annotations from the dataframe.
    """

    s3 = boto3.client('s3')
    s3_bucket_name = "celiaproject"

    # print all the videos with their status side by side
    biigle_videos = all_biigle_df.video_filename.unique().tolist()
    s3_videos = [f["Key"] for f in s3.list_objects(Bucket=s3_bucket_name)["Contents"]]

    print(tabulate.tabulate(
        [(i, v, v in s3_videos) for i, v in enumerate(biigle_videos)],
        headers=["index", "video_filename", "video_exists"]
    ))

    all_biigle_df.loc[:, 'video_exists'] = all_biigle_df.video_filename.apply(lambda x: x in s3_videos)

    # remove videos that do not exist
    all_biigle_df = all_biigle_df[all_biigle_df.video_exists == True]

    return all_biigle_df


def train_val_split(all_biigle_df, out_dir, train_ratio=0.7, split_by="video", seed=42):
    """
    Split the data into train and validation sets.
    """
    # set the random seed
    random.seed(seed)
    
    class_labels = all_biigle_df.label_name.unique().tolist()
    # sort the class labels by their counts
    class_labels.sort(key=lambda x: all_biigle_df[all_biigle_df.label_name == x].shape[0], reverse=True)

    # write the class labels to a file
    with open(os.path.join(out_dir, "metadata", "class_labels.txt"), "w") as f:
        f.write('\n'.join(class_labels))
    

    video_ids = all_biigle_df.video_id.unique().tolist()
    video_ids.sort(key=lambda x: all_biigle_df[all_biigle_df.video_id == x].shape[0], reverse=True)
    # print the number of unique videos with their counts
    print("Number of unique videos: ", len(video_ids))
    print(tabulate.tabulate(
        [(i, v, all_biigle_df[all_biigle_df.video_id == v].shape[0]) for i, v in enumerate(video_ids)],
        headers=["index", "video_id", "frame count"]
    ))
    print("Number of unique labels: ", len(class_labels))
    print(tabulate.tabulate(
        [(i, c, all_biigle_df[all_biigle_df.label_name == c].shape[0]) for i, c in enumerate(class_labels)],
        headers=["index", "category", "count"]
    ))
    print("Total number of annotations: ", all_biigle_df.shape[0])
    print("Total number of frames: ", len(all_biigle_df.frame_id.unique()))

    if split_by == "video":
        # We will use the first X% of the videos for training and the rest for validation
        train_video_ids = video_ids[:int(len(video_ids) * train_ratio)]
        val_video_ids = video_ids[int(len(video_ids) * train_ratio):]

        # split the dataframes
        train_df = all_biigle_df[all_biigle_df.video_id.isin(train_video_ids)]
        val_df = all_biigle_df[all_biigle_df.video_id.isin(val_video_ids)]
    elif split_by == "frame":
        # We will use the X% of the frames for training and the rest for validation
        # but we need to shuffle the frames first 
        # to avoid having all the frames of a video in the same set
        frame_ids = all_biigle_df.frame_id.unique().tolist()
        random.shuffle(frame_ids)

        # split the dataframes
        train_frame_ids = frame_ids[:int(len(frame_ids) * train_ratio)]
        val_frame_ids = frame_ids[int(len(frame_ids) * train_ratio):]

        train_df = all_biigle_df[all_biigle_df.frame_id.isin(train_frame_ids)]
        val_df = all_biigle_df[all_biigle_df.frame_id.isin(val_frame_ids)]
    else:
        raise ValueError(f"Unknown split_by value: {split_by}")
    
    # print the number of unique labels with their counts
    # in the train and validation sets
    print("Number of unique labels in training set: ", len(train_df.label_name.unique()))
    print_class_counts(train_df)
    print("Number of unique labels in validation set: ", len(val_df.label_name.unique()))
    print_class_counts(val_df)
    

    # write the dataframes to csv files
    train_df.to_csv(os.path.join(out_dir, "metadata", "train_biigle.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "metadata", "val_biigle.csv"), index=False)

    # convert the dataframes to detectron datasets
    frames_root_dir = os.path.join(out_dir, "frames", "labeled")
    train_dataset = biigle_dataframe_to_detectron_dataset(train_df, frames_root_dir, class_labels)
    val_dataset = biigle_dataframe_to_detectron_dataset(val_df, frames_root_dir, class_labels)

    # write the datasets to json files
    train_dataset_filename = os.path.join(out_dir, "metadata", "train_detectron.json")
    val_dataset_filename = os.path.join(out_dir, "metadata", "val_detectron.json")
    with open(train_dataset_filename, "w") as f:
        json.dump(train_dataset, f)
    with open(val_dataset_filename, "w") as f:
        json.dump(val_dataset, f)
    print("Saved train dataset to ", train_dataset_filename)
    print("Saved val dataset to ", val_dataset_filename)



def extract_frames(all_biigle_df, out_dir):
    """
    Extract frames from the videos and save them to disk.
    """
    video_dir = os.path.join(out_dir, "videos")
    download_mp4_if_not_exists(all_biigle_df, video_dir)

    for video_id in all_biigle_df.video_id.unique():
        print(f"Processing video {video_id}...")
        single_video_df = all_biigle_df[all_biigle_df.video_id == video_id]
        video_filename = single_video_df.video_filename.unique()[0]
        video_path = os.path.join(video_dir, video_filename)

        label_timestamps = single_video_df.frame.unique().tolist()
        frames_dir = os.path.join(out_dir, "frames", "labeled", str(video_id))

        os.makedirs(frames_dir, exist_ok=True)

        print(f"Extracting frames from video {video_path}...")
        # extract frames from video
        ts_to_fnames = {ts: f"{video_id}_{ts}.jpg" for ts in label_timestamps}
        ts_to_frames = extract_frames_from_video(video_path, label_timestamps)
        fnames_to_frames = {ts_to_fnames[ts]: frame for ts, frame in ts_to_frames.items()}

        print(f"Saving frames to {frames_dir}...")
        # save frames to disk
        save_frames_to_disk(fnames_to_frames, frames_dir)


def gather_biigle_csv(biigle_csv_dir):
    """
    Gather all the biigle csv files into a single pandas dataframe
    and clean up the data
    """

    csv_files = glob.glob(os.path.join(biigle_csv_dir, '*.csv'))
    all_df = pd.DataFrame()
    for csv_file in csv_files:
        video_df = pd.read_csv(csv_file)
        all_df = pd.concat([all_df, video_df])

    # replace all "+" in filenames with " " to 
    # avoid issues with S3
    all_df.loc[:, 'video_filename'] = all_df.video_filename.apply(lambda x: x.replace("+", " "))

    # remove videos that do not exist
    all_df = remove_unavailable_videos(all_df)

    # fix the frames column
    all_df.loc[:, 'frames'] = all_df.frames.apply(lambda x: json.loads(x))

    # if there are multiple frames per row, split them into multiple rows
    all_df = all_df.explode('frames')
    # rename the 'frames' column to 'frame'
    all_df.rename(columns={'frames': 'frame'}, inplace=True)

    # create frame_id column by concatenating video_id and frame
    frame_ids = all_df.video_id.astype(str) + "_" + all_df.frame.astype(str)
    all_df.loc[:, 'frame_id'] = frame_ids

    return all_df


def main(biigle_csv_dir, output_dir, train_ratio, skip_extract_frames):
    """
    Main function.

    Directory structure:
    out_dir
    ├── frames
    │  ├── labeled
    │  │  ├── video_id_1
    │  │  │  ├── frame_1.jpg
    │  │  │  ├── frame_2.jpg
    │  │  │  ├── ...
    │  │  │  └── frame_n.jpg
    │  │  ├── video_id_2
    │  │  │  ├── ...
    │  │  │  └── frame_n.jpg
    │  │  ├── ...
    │  │  └── video_id_n
    │  │      ├── ...
    │  │      └── frame_n.jpg
    │  └── unlabeled
    │      ├── ...
    ├── videos
    │  ├── video_id_1.mp4
    │  ├── video_id_2.mp4
    │  ├── ...
    │  └── video_id_n.mp4
    └── metadata
        ├── all_biigle.csv
        ├── train_biigle.csv
        ├── val_biigle.csv
        ├── train_detectron.json
        └── val_detectron.json
        

    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames", "labeled"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames", "unlabeled"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

    # Gather all the biigle csv files into a single pandas dataframe
    print("Gathering biigle csv files...")
    all_df = gather_biigle_csv(biigle_csv_dir)
    all_df_filename = os.path.join(output_dir, "metadata", "all_biigle.csv")
    all_df.to_csv(all_df_filename, index=False)
    print("Saved all_biigle.csv to ", all_df_filename)

    # Create the train/val split metadata files
    print("Creating metadata files...")
    train_val_split(all_df, output_dir, split_by="frame", train_ratio=train_ratio)

    if skip_extract_frames:
        return
    
    # Then extract the frames
    print("Extracting frames...")
    extract_frames(all_df, output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--biigle-csv-dir", help="Path to the biigle csv files", required=True)
    parser.add_argument("--output-dir", help="Output directory", required=True)
    parser.add_argument("--train-ratio", help="Percentage of the data to use for training", default=0.95, type=float)
    parser.add_argument("--skip-extract-frames", help="Skip extracting frames from videos", action="store_true")
    args = parser.parse_args()

    main(
        biigle_csv_dir=os.path.expanduser(args.biigle_csv_dir), 
        output_dir=os.path.expanduser(args.output_dir),
        train_ratio=args.train_ratio,
        skip_extract_frames=args.skip_extract_frames,
    )
    

