from transformers import AutoProcessor, LlavaForConditionalGeneration
from .Captioner import Captioner
from PIL import Image
import torch


def parse_llava_output(output: str) -> str:
    # Find the position of "ASSISTANT:"
    assistant_pos = output.find("ASSISTANT:")
    
    # If found, return everything after "ASSISTANT:", otherwise return original string
    if assistant_pos != -1:
        # Add len("ASSISTANT:") to skip over the marker itself
        return output[assistant_pos + len("ASSISTANT:"):].strip()
    return output.strip()

class LlavaCaptioner(Captioner):
    def __init__(self,prompt,llava_model,device=None):
        self.device=device
        if not device:
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
        return parse_llava_output(self.processor.decode(output[0][2:], skip_special_tokens=True))
    
if __name__=="__main__":
    llava=LlavaCaptioner("Caption this image")
    llava.caption("misc/test.jpg")   