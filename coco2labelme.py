"""
Convert biigle annotation format to labelme format
"""
import pandas as pd
import tabulate
import json
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Convert biigle annotation format to labelme format')
parser.add_argument('biigle', type=str, help='Path to biigle COCO file')
parser.add_argument('--output-directory', type=str, default="/tmp/", help='Path to output directory')
args = parser.parse_args()


biigle_csv = args.biigle
output_directory = args.output_directory

# The biigle format is a CSV file with the following headers:
# video_annotation_label_id,label_id,label_name,label_hierarchy,user_id,firstname,lastname,video_id,video_filename,shape_id,shape_name,points,frames,annotation_id,created_at

# One line of the CSV file looks like this:
# 1475726,222120,Other,"Biota > Sponges > Massive > Other",2156,Celia,Hanna,16365,file4dive8jc66_1.mp4,5,Rectangle,"[[679.24,246.12,732.77,408.79,595.3,454.03,541.76,291.37]]",[6015.125313107339],1253045,"2023-06-05 16:21:15"


# One file of the labelme format looks like this:
#{
#   "version": "5.0.1",
#   "flags": {},
#   "shapes": [
#     {
#       "label": "ship",
#       "points": [
#         [
#           239.0990099009901,
#           420.2970297029703
#         ],
#         [
#           423.25742574257424,
#           338.1188118811881
#         ],
#         [
#           444.54455445544556,
#           345.049504950495
#         ],
#         [
#           434.64356435643566,
#           365.84158415841586
#         ],
#         [
#           253.45544554455444,
#           446.53465346534654
#         ]
#       ],
#       "group_id": null,
#       "shape_type": "polygon",
#       "flags": {}
#     }
#   ],
#   "imagePath": "1.png",
#   "imageData": null,
#   "imageHeight": 480,
#   "imageWidth": 640
# }

data = pd.read_csv(biigle_csv)

# assert each row has a single frame


# fix the frames column
data.frames = data.frames.apply(lambda x: json.loads(x))

# assert data belongs to a single video
assert data.video_id.nunique() == 1


# The labelme format requires a list of categories. We will create a list of unique
# categories from the biigle data.
categories = data.label_name.unique().tolist()
categories.sort()
# print a table of categories, indices and their counts
print(tabulate.tabulate(
    [(i, c, data[data.label_name == c].shape[0]) for i, c in enumerate(categories)],
    headers=["index", "category", "count"]
))

# print a python list of categories
print(f"categories = {categories}")

# The labelme format generates one json file per image.
labelme_data = {}

for i, row in data.iterrows():
    frames = row.frames
    for frame in frames:
        shapes = []
        # create a json object for each annotation
        per_frame_points = json.loads(row.points)
        
        for points in per_frame_points:
            points = [[p1, p2] for p1, p2 in zip(points[::2], points[1::2])]
            shape = {
                "label": row.label_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)


        annotation = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": f"{frame}.jpg",
            "imageData": None,
            "imageHeight": 1080,
            "imageWidth": 1920
        }
    
        if frame in labelme_data:
            labelme_data[frame]["shapes"].extend(shapes)
        else:
            labelme_data[frame] = annotation

# save the labelme data to json files
for frame, annotation in labelme_data.items():
    with open(os.path.join(output_directory, f"{frame}.json"), "w") as f:
        json.dump(annotation, f, indent=2)

print(f"Saved {len(labelme_data)} json files to {output_directory}")


# print those frames that have more than one annotation
# print("Frames with more than one annotation:")
# for frame, annotations in labelme_data.items():
#     if len(annotations["shapes"]) > 1:
#         print(f"Frame {frame} has more than one annotation")











