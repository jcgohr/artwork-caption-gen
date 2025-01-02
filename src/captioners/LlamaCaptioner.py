from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
from Captioner import Captioner
from dotenv import load_dotenv
from PIL import Image
import torch
import os


class LlamaCaptioner(Captioner):
    def __init__(self,model_id:str,env_path=".env"):
        
        # Login to hugging face to get access to Llama models
        load_dotenv(env_path)
        login(os.getenv("HF_TOKEN"))
        
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def caption(self, image_path):
        image = Image.open(image_path)

        prompt = "<|image|><|begin_of_text|> Write a caption for this image"
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=30)
        print(self.processor.decode(output[0]))
        
        
if __name__=="__main__":
    captioner=LlamaCaptioner("meta-llama/Llama-3.2-11B-Vision")
    captioner.caption("misc/test.jpg")