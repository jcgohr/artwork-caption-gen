from lavis.models import load_model_and_preprocess
from Captioner import Captioner
from PIL import Image
import torch



class BlipCaptioner(Captioner):
    def __init__(self):
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
        # this also loads the associated image processors
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True, device=self.device)
    
    def caption(self, image_path):
        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = self.vis_processors["eval"](Image.open(image_path)).unsqueeze(0).to(self.device)
        # generate caption
        return self.model.generate({"image": image})
        # ['a large fountain spewing water into the air']
        
if __name__=="__main__":
    blipper=BlipCaptioner()
    print(blipper.caption("misc/test.jpg"))