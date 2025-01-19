from parsers import CaptionGenerationParser 
from captioners import *
from tqdm import tqdm 
import torch
import json
import os


if __name__=="__main__":
    captioners=globals()
    parser=CaptionGenerationParser()


    args=parser.parse_args()    

    confs=args.config
    data=args.metadata
    output=args.output

    captions={}
    params=list(confs.values())
    captioners=list(confs.keys())

    captions=multi_gpu_captioning(captioners,params,data)
    with open(output,"w",encoding="utf-8") as output_file:
        output_file.write(json.dumps(captions,indent=4,ensure_ascii=False))