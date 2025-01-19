from .utils import get_captioners,split_dict
from multiprocessing import Pool
from tqdm import tqdm
import torch
import json

class Captioner:
    def caption(self,image_path:str)->str:
        raise NotImplemented


FP_KEY="file_path"



def sequential_captioning(captioners:list[str],params:list[list],data:dict,device:str):
    captions={}
    classes=get_captioners()
    for captioner,param in zip(captioners,params):
        model=classes[captioner](*param,device=device)
        for key in tqdm(data,desc=f"Generating captions with {captioner}"):
            if key not in captions:
                captions[key]={}
            captions[key][captioner]=model.caption(data[key][FP_KEY])

    return captions
    
def multi_gpu_captioning(captioners:list[str],params:list[list],data:dict):
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available")
    
    # gpus = torch.cuda.device_count()
    gpus=2
    if gpus==1:
        raise ValueError("Only 1 GPU detected, try Captioner.sequential_captioning instead")
    
    devices=[f"cuda:{i}" for i in range(gpus)]
    subsets = split_dict(data,gpus)
    for captioner,param in zip(captioners,params):
        # Make variables compatible with sequential_captioning
        captioner=[[captioner]]*gpus
        param=[[param]]*gpus
        map_params=[[a,b,c,d] for a,b,c,d in zip(captioner,param,subsets,devices)]
        with Pool(gpus) as pool:
            for result in pool.starmap(sequential_captioning,map_params):
                print(result)
