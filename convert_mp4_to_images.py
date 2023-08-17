import json
import argparse
import os
import cv2
import tqdm
import pandas as pd
import concurrent.futures
import numpy as np

def extract_frame(video_path, frame_ids):
    frames = {}
    cap = cv2.VideoCapture(video_path)
    filename = os.path.basename(video_path)
    for frame_id in tqdm.tqdm(
        frame_ids, 
        desc=f"Extracting frames from {filename} @ frame=[{min(frame_ids)}:{max(frame_ids)}]", 
        leave=False
    ):
        # set the video to the target frame (this just skips frames until we get to the target frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id-1)
        ret, frame = cap.read()
        
        while not ret:
            raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")
            # print("Frame not extracted, trying again...")
            # ret, frame = cap.read()
        
        # double check that we are at the right frame
        curr_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if abs(curr_frame_id - frame_id) > 10:
            print(f"Found the next frame for {frame_id},\
                but it is more than 10 frames away curr_frame_id: {curr_frame_id}")
            continue

        frames[frame_id] = frame

    cap.release()
    return frames
    
def extract_frames_from_video(video_path, timestamps):

    cap = cv2.VideoCapture(video_path)
    frames = {}
    
    # Convert timestamps to frame indices
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    # frame_indices = [int(timestamp * fps) for timestamp in timestamps]
    frame_to_ts = {timestamp * fps: timestamp for timestamp in timestamps}
    ts_to_frame = {timestamp: timestamp * fps for timestamp in timestamps}

    
    print(f"Extracting frames from {video_path} at timestamps {timestamps}...")
    print(f"Video FPS: {fps}")
    print(f"ts_to_frame: {ts_to_frame}")
    print("\n\n")
    print(f"frame_to_ts: {frame_to_ts}")
    
    assert len(ts_to_frame) == len(frame_to_ts), "ts_to_frame and frame_to_ts must have the same length"
    assert all([frame_id in frame_to_ts for frame_id in ts_to_frame.values()]), "frame_to_ts must have all frame_ids in ts_to_frame.values()"
    assert all([frame_id in ts_to_frame.values() for frame_id in frame_to_ts.keys()]), "ts_to_frame must have all frame_ids in frame_to_ts.keys()"

    
    
    # use ThreadPoolExecutor to extract frames in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        target_frames = list(frame_to_ts.keys())
        chunk_size = 10
        num_chunks = len(target_frames) // chunk_size
        ceil = 1 if len(target_frames) % chunk_size > 0 else 0
        num_chunks += ceil
        with tqdm.tqdm(total=num_chunks, desc=f"Extracting frames from {video_path}") as pbar:
            for i in range(0, len(target_frames), chunk_size):
                chunk = target_frames[i:i+chunk_size]
                future = executor.submit(extract_frame, video_path, chunk)
                futures.append(future)
                future.add_done_callback(lambda x: pbar.update(1))
                
            # futures = [executor.submit(extract_frame, video_path, target_frame) for target_frame in frame_to_ts.keys()]
            for future in futures:
                worker_frames = future.result()
                for frame_id, frame in worker_frames.items():
                    ts = frame_to_ts[frame_id]
                    frames[ts] = frame


    # Sequential version
    # for target_frame in tqdm.tqdm(frame_indices):
    #     # set the video to the target frame (this just skips frames until we get to the target frame)
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame-1)
    #     ret, frame = cap.read()
        
    #     if not ret:
    #         break
        
    #     # double check that we are at the right frame
    #     frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
    #     if frame_id in frame_indices:
    #         frames.append(frame)
    #     else:
    #         print(f"Frame {frame_id} is not in frame_indices.")
            
    #     if len(frames) == len(frame_indices):
    #         break
    # cap.release()

    return frames

def save_frames_to_disk(fnames_to_frames, output_directory):

    assert len(fnames_to_frames) > 0, "No frames to save"
    assert isinstance(fnames_to_frames, dict), "fnames_to_frames must be a dictionary"
    assert all([fname.endswith(".jpg") for fname in fnames_to_frames.keys()]), "All filenames must end with .jpg"
    assert all([isinstance(frame, np.ndarray) for frame in fnames_to_frames.values()]), "All values must be numpy arrays"

    print(f"Saving {len(fnames_to_frames)} frames to {output_directory}...")
    os.makedirs(output_directory, exist_ok=True)

    for fname, frame in tqdm.tqdm(fnames_to_frames.items()):
        output_path = os.path.join(output_directory, fname)
        ret = cv2.imwrite(output_path, frame)
        if not ret:
            raise RuntimeError(f"Could not write frame {fname} to {output_path}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="""
Extract frames from a video. 
If --timestamps is provided, extract frames at those timestamps.
If --biigle-csv is provided, extract frames at the timestamps in the csv file.

Example usage:
python extract_frames_from_video.py --video_path /path/to/video.mp4 --timestamps 0 10 --output_directory /path/to/output/directory
python extract_frames_from_video.py --video_path /path/to/video.mp4 --biigle-csv /path/to/biigle.csv --output_directory /path/to/output/directory


Use --timedelta  to extract frames every n seconds (e.g. every 10 seconds)
Example usage with timedelta:
python extract_frames_from_video.py --video_path /path/to/video.mp4 --timestamps 0 10 --output_directory /path/to/output/directory --timedelta 10.0
""")
    args.add_argument("--video_path", type=str, required=True)
    args.add_argument("--timestamps", help="From and to timestamps (in sec). Exclusive with --biigle", nargs="+", type=float)
    args.add_argument("--biigle-csv", help="get timestamps of keyframes from biigle csv. Exclusive with --timestamps", type=str)
    args.add_argument("--output_directory", type=str, default="/tmp/", help="directory to save extracted frames")
    args.add_argument("--timedelta", type=float, help="extract frames every n seconds (e.g. 0.5). Default: extract every frame")
    args = args.parse_args()

    video_path = args.video_path

    if args.timedelta is None:
        fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
        args.timedelta = 1/fps

    if args.timestamps is not None:
        timestamps_from = args.timestamps[0]
        timestamps_to = args.timestamps[1]
        timestamps = list(np.arange(timestamps_from, timestamps_to, args.timedelta))

    if args.timestamps is None:
        timestamps = pd.read_csv(args.biigle_csv)["frames"]
        timestamps = [json.loads(ts) for ts in timestamps]
        # flatten the list
        timestamps = [item for sublist in timestamps for item in sublist]

    timestamps = list(set(timestamps))
    timestamps = sorted(timestamps)

    output_directory = args.output_directory
    extracted_frames = extract_frames_from_video(video_path, timestamps)
    save_frames_to_disk(extracted_frames, output_directory)

