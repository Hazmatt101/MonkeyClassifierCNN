from PIL import Image
from resizeimage import resizeimage
import os
import re

for i in range(0, 10):
    directory = os.fsencode("./10-monkey-species/validation/n{}/".format(i))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if filename != '.DS_Store':
            open_temp_path = ("./10-monkey-species/validation/n{}/".format(i)) + filename
            print(open_temp_path)

            with open(open_temp_path, 'r+b') as f:
                with Image.open(f) as image:
                    cover = resizeimage.resize_cover(image, [300, 300])
                    filename = re.sub('.jpg', '', filename)
                    filename = filename + '-resized.jpg'
                    write_temp_path = ("./validation-resized/n{}/".format(i)) + filename
                    cover.save(write_temp_path, image.format)
