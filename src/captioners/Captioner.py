from tqdm import tqdm
import importlib
import inspect
import torch

class Captioner:
    def caption(self,image_path:str)->str:
        raise NotImplemented


FP_KEY="file_path"

def get_captioners():
    module = importlib.import_module('.', package='captioners')
    classes = inspect.getmembers(module, inspect.isclass)
    return {k:v for k,v in classes}

def sequential_captioning(captioners:list[str],params:list[list],data:dict):
    captions={}
    classes=get_captioners()
    for captioner,param in zip(captioners,params):
        model=classes[captioner](*param)
        for key in tqdm(data,desc=f"Generating captions with {captioner}"):
            if key not in captions:
                captions[key]={}
            captions[key][captioner]=model.caption(data[key][FP_KEY])

    return captions
    
def multi_gpu_captioning(captioners:list[str],params:list[list],data:dict):
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available")
    
    gpus = torch.cuda.device_count()
    if gpus==1:
        raise ValueError("Only 1 GPU detected, try Captioner.sequential_captioning instead")
    
if __name__=="__main__":
    multi_gpu_captioning([],[],{})