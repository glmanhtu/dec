import os
import sys
from PIL import Image

w, h = (200, 120)

if __name__ == '__main__':
    images_dir = sys.argv[1]
    if os.path.isdir(images_dir):
        images_dir = os.path.abspath(images_dir)
        for root, dirs, files in os.walk(images_dir):
            for f in files:
                image = os.path.join(root, f)
                print image
                im = Image.open(image).resize((w, h), Image.BICUBIC)
                im.save(image)

    else:
        raise Exception("Please specific image url")