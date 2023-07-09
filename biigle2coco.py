"""
Convert biigle annotation format to COCO
"""
import pandas as pd
import tabulate
import json
import sys
import os

biigle_csv = sys.argv[1]
coco_json = os.path.splitext(biigle_csv)[0] + ".json"

# The biigle format is a CSV file with the following headers:
# video_annotation_label_id,label_id,label_name,label_hierarchy,user_id,firstname,lastname,video_id,video_filename,shape_id,shape_name,points,frames,annotation_id,created_at

# One line of the CSV file looks like this:
# 1475726,222120,Other,"Biota > Sponges > Massive > Other",2156,Celia,Hanna,16365,file4dive8jc66_1.mp4,5,Rectangle,"[[679.24,246.12,732.77,408.79,595.3,454.03,541.76,291.37]]",[6015.125313107339],1253045,"2023-06-05 16:21:15"

# The COCO format is described here: https://cocodataset.org/#format-data
# The COCO format is a JSON file with the following structure:
# {
#     "info": info,
#     "licenses": [license],
#     "images": [image],
#     "annotations": [annotation],
#     "categories": [category]
# }
#

data = pd.read_csv(biigle_csv)

# assert data belongs to a single video
assert data.video_id.nunique() == 1


# The COCO format requires a list of categories. We will create a list of unique
# categories from the biigle data.
categories = data.label_name.unique().tolist()
categories.sort()
# print a table of categories, indices and their counts
print(tabulate.tabulate(
    [(i, c, data[data.label_name == c].shape[0]) for i, c in enumerate(categories)],
    headers=["index", "category", "count"]
))

# The COCO format requires a list of images.
# Since one annotation csv file contains only one video, 
# we will list the frames of the video as images.
images = data.frames.unique().tolist()

# The COCO format requires a list of annotations.
# We will create a list of annotations from the biigle data.
annotations = []
for i, row in data.iterrows():
    # check for empty points
    if row.points == "[]":
        # this is a whole frame annotation (treat as iscrowd=1)
        annotations.append({
            "id": row.annotation_id,
            "image_id": row.frames,
            "category_id": categories.index(row.label_name),
            "bbox": [0, 0, 0, 0],
            "area": 0,
            "iscrowd": 1,
        })
        continue

    biigle_bbox = json.loads(row.points)[0]
    # bbox is encoded as x1, y1, x2, y2, x3, y3, x4, y4
    # we will convert it to COCO format: x1, y1, width, height
    x1, y1, x2, y2, x3, y3, x4, y4 = biigle_bbox

    # check if the rectangle is rotated
    # the number of unique x coordinates should be:
    # 2 if the rectangle is not rotated
    # 4 if the rectangle is rotated
    if len(set([x1, x2, x3, x4])) == 4:
        # rotated rectangle
        # we will convert it to a bounding box
        x1 = min([x1, x2, x3, x4])
        x2 = max([x1, x2, x3, x4])
        y1 = min([y1, y2, y3, y4])
        y3 = max([y1, y2, y3, y4])

    width = x2 - x1
    height = y3 - y1

    coco_bbox = [x1, y1, width, height]
    
    annotations.append({
        "id": row.annotation_id,
        "image_id": row.frames,
        "category_id": categories.index(row.label_name),
        "bbox": coco_bbox,
        "area": width * height,
        "iscrowd": 0,
    })

# Save the COCO format to a JSON file
coco = {
    "info": {
        "description": "Converted from biigle format",
        "video": data.video_filename.unique().tolist()[0],
    },
    "licenses": [],
    "images": [{"id": i} for i in images],
    "annotations": annotations,
    "categories": [{"id": i, "name": c} for i, c in enumerate(categories)],
}













