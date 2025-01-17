from captioners import * 
from parsers import CaptionGenerationParser  
import os

"""
Step 1: Parse a json file to get the file paths
Step 2: Open the file and caption the image
Step 3: Store the caption into a captions json file     

Questions:
Should all captions be stored in the same file?
Should we make a new caption file for every run of this script?

"""

FP_KEY="file_path"
captioners=globals()
parser=CaptionGenerationParser()


args=parser.parse_args()

confs=args.config
data=args.metadata
output=args.output

captions={}
for class_name in confs:
    model=captioners[class_name](*confs[class_name])
    for key in data:
        captions[key][class_name]=model.caption(data[key][FP_KEY])
