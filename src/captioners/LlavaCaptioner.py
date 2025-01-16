from transformers import AutoProcessor, LlavaForConditionalGeneration
from .Captioner import Captioner
from PIL import Image
import torch


class LlavaCaptioner(Captioner):
    def __init__(self,prompt,llava_model="llava-hf/llava-1.5-7b-hf"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            llava_model, 
            torch_dtype=torch.float16, 
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(llava_model)
        
        self.conversation = [
            {

            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
               
                ],
            },
        ]
        self.prompt = self.processor.apply_chat_template(self.conversation, add_generation_prompt=True)
    
    def caption(self, image_path):
        raw_image = Image.open(image_path)
        inputs = self.processor(images=raw_image, text=self.prompt, return_tensors='pt').to(self.device, torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        return self.processor.decode(output[0][2:], skip_special_tokens=True)
    
if __name__=="__main__":
    llava=LlavaCaptioner("Caption this image")
    llava.caption("misc/test.jpg")   