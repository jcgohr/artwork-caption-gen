from parsers import CaptionGenerationParser  
from captioners import *
from tqdm import tqdm 
import json
import os


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
    for key in tqdm(data,desc=f"Generating captions with {class_name}"):
        if key not in captions:
            captions[key]={}
        captions[key][class_name]=model.caption(data[key][FP_KEY])

with open(output,"w",encoding="utf-8") as output_file:
    output_file.write(json.dumps(captions,indent=4,ensure_ascii=False))