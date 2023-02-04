#!/usr/bin/env python3 

# This script is for reading a PDF of Homestead claims to locate:
#  * Name, age, and place of birth
#  * Location of claimed homestead
#  * acreage of homestead claim
#  * dates (at least their entry date and date that patent was issued)
#  * value of improvements, itemized
#  * crop plantings, including acreage, crop selection, and harvest
#  * witnesses
# These typically appear in the same places on each page, so we can use
# a template to find them. The values are not typically OCRed very well, 
# we'll have to use computer vision to find them. 

import sys
import os
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image as PILImage
from pdf2image import convert_from_path

# This is the template we'll use to find the values we want
template = cv2.imread('template.png', 0)

# This is the PDF we'll read
pdf = sys.argv[1]

# This is the directory we'll write the images to
image_dir = sys.argv[2]

# This is the directory we'll write the output to
output_dir = sys.argv[3]

# First, convert the PDF to images
images = convert_from_path(pdf, 500)

# Now, for each image, find the template
for i, image in enumerate(images):
    # Convert the image to grayscale
    image = image.convert('L')
    # Save the image
    image.save(os.path.join(image_dir, f'{i}.png'))
    # Convert the image to a numpy array
    image = np.array(image)
    # Find the template in the image
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # Get the location of the template
    loc = np.where(res >= 0.8)
    # Get the location of the template
    for pt in zip(*loc[::-1]):
        # Crop the image to the template
        crop = image[pt[1]:pt[1]+template.shape[0], pt[0]:pt[0]+template.shape[1]]
        # Convert the crop to a PIL image
        crop = PILImage.fromarray(crop)
        # Save the crop
        crop.save(os.path.join(output_dir, f'{i}.png'))
        # Read the text from the crop
        text = pytesseract.image_to_string(crop)
        # Write the text to a file
        with open(os.path.join(output_dir, f'{i}.txt'), 'w') as f:
            f.write(text)

# Now, we can read the text from the crops and parse it
# This is a list of the fields we want to extract
fields = [
    'Name',
    'Age',
    'Place of Birth',
    'Location of Claim',
    'Acreage',
    'Entry Date',
    'Patent Date',
    'Improvements',
    'Crop Plantings',
    'Witnesses',
]

# This is a list of the fields we want to extract
field_regexes = [
    r'Name: (.*)',
    r'Age: (.*)',
    r'Place of Birth: (.*)',
    r'Location of Claim: (.*)',
    r'Acreage: (.*)',
    r'Entry Date: (.*)',
    r'Patent Date: (.*)',
    r'Improvements: (.*)',
    r'Crop Plantings: (.*)',
    r'Witnesses: (.*)',
]

# now, for each text file, extract the fields
for i, filename in enumerate(os.listdir(output_dir)):
    # skip the template
    if filename == 'template.png':
        continue
    # read the text
    with open(os.path.join(output_dir, filename), 'r') as f:
        text = f.read()
    # extract the fields
    for field, field_regex in zip(fields, field_regexes):
        # find the field
        match = re.search(field_regex, text)
        # if we found the field
        if match:
            # print the field
            print(f'{field}: {match.group(1)}')
        # otherwise
        else:
            # print the field
            print(f'{field}:')
    # print a blank line
    print()
