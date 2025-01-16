import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def classify(sentence: str)->bool:
    input = tokenizer(sentence, return_tensors="pt").to(model.device)
    model.generate
    pipe = pipeline(
        "text-generation", 
        model=MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

pipe("The key to life is")