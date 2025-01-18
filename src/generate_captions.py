from parsers import CaptionGenerationParser 
from captioners import *
from tqdm import tqdm 
import json
import os



captioners=globals()
parser=CaptionGenerationParser()


args=parser.parse_args()

confs=args.config
data=args.metadata
output=args.output

captions={}
params=list(confs.values())
captioners=list(confs.keys())


captions=sequential_captioning(captioners,params,data)
with open(output,"w",encoding="utf-8") as output_file:
    output_file.write(json.dumps(captions,indent=4,ensure_ascii=False))