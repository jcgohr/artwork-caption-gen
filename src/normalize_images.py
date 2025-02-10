from custom_datasets import normalize_image
import sys
import os

image_directory=sys.argv[1]

for image in os.listdir(image_directory):
    image_path=os.path.join(image_directory,image)
    normalize_image(image_path)