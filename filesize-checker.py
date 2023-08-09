# go through all images and check their height and width
# log all unique sizes and their count

import os
import glob
import imagesize
import tqdm
from collections import defaultdict

# get all images
images = glob.glob('../../storage/deepsea/frames/labeled/**/*.jpg', recursive=True)

# get all unique sizes
sizes = defaultdict(list)
for image in tqdm.tqdm(images):
    width, height = imagesize.get(image)
    sizes[(width, height)].append(image)
    

# print all unique sizes
del sizes[(1920, 1080)]
print(sizes)