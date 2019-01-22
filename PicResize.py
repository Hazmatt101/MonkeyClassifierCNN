from PIL import Image
from resizeimage import resizeimage
import os
import re

directory = os.fsencode("./10-monkey-species/training/n0/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)

    with open(filename, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [400, 300])
            filename = re.sub('.jpg', '', filename)
            filename = filename + '-resized.jpg'
            cover.save(filename, image.format)
