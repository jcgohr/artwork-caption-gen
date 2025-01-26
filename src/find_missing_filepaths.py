import json
import sys
import os

fp=sys.argv[1]
output=sys.argv[2]

with open(fp,"r",encoding="utf-8") as f:
    artpedia=json.load(f)
    
    
missing_items={}
for key in artpedia:
    if "file_path" not in artpedia[key]:
        missing_items[key]=artpedia[key]
        
with open(output,"w",encoding="utf-8") as o:
    o.write(json.dumps(missing_items,indent=4,ensure_ascii=False))