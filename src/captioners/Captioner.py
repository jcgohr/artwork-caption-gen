import torch

class Captioner:
    def caption(self,image_path:str)->str:
        raise NotImplemented
    
    
    
def multi_gpu_captioning(captioners:list[str],params:list[list],data:dict):
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available")
    
    gpus = torch.cuda.device_count()
    if gpus==1:
        raise ValueError("Only 1 GPU detected, try Captioner.sequential_captioning instead")
    
if __name__=="__main__":
    multi_gpu_captioning([],[],{})