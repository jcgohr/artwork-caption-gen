from transformers import MllamaForConditionalGeneration, MllamaProcessor
from huggingface_hub import login
from .Captioner import Captioner
from dotenv import load_dotenv
from PIL import Image
import torch
import os


class LlamaCaptioner(Captioner):
    def __init__(self,model_id:str,prompt:str,env_path=".env",device=None):
        
        self.device=device
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Login to hugging face to get access to Llama models
        load_dotenv(env_path)
        login(os.getenv("HF_TOKEN"))
        
  
            
            
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        try:
            self.model.to(device)
        except:
            print("Model was not successfully loaded to GPU")
        
        self.prompt=prompt
        self.processor = MllamaProcessor.from_pretrained(model_id)
        
        
        
    def caption(self, image_path):
        image = Image.open(image_path)

        prompt = f"<|image|><|begin_of_text|>{self.prompt}"
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=75)
        prompt_len = inputs.input_ids.shape[-1]

        generated_ids = output[:, prompt_len:]

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return generated_text[0]
        
        
if __name__=="__main__":
    captioner=LlamaCaptioner("meta-llama/Llama-3.2-11B-Vision","Caption this image")
    captioner.caption("misc/test.jpg")