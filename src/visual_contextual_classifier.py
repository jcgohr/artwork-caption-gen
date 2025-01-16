import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_PROMPT = [
    {"role": "system",
        "content": (
        "You are a sentence classifier that is given a sentence by the user which describes a painting, and you output a '1' or '0' based on the following criteria: "
        "You output 1 if the sentence primarily describes visual content within the painting. "
        "You output 0 if the sentence primarily describes contextual information about the painting.")},

    {"role": "user", 
        "content": "<insert sentence>"}
]
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

class classifier:
    """
    Classify sentences as visual or contextual with an LM
    """
    def __init__(self, model_id:str, prompt:str=DEFAULT_PROMPT, **kwargs):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=ACCESS_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=ACCESS_TOKEN,**kwargs)
        self.prompt = prompt

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def classify(self, sentences:str)->list[bool]:
        """
        Classify a list of sentences as either 1 (visual), or 0 (contextual)
        """
        formatted_prompt = self._sentences_to_prompt(sentences)

        # Forward pass
        inputs = self.tokenizer(formatted_prompt, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get classifications from logits
        id_1 = self.tokenizer.convert_tokens_to_ids("1")
        id_0 = self.tokenizer.convert_tokens_to_ids("0")
        last_logits = logits[:,-1,:] # shape: (batch_size, 1, vocab_size)
        # for sentence_logits in last_logits.squeeze(1):
        #     print(sentence_logits[id_1])
        #     print(sentence_logits[id_0])
        #     print(sentence_logits[id_1]>sentence_logits[id_0])
        predictions = [sentence_logits[id_1] > sentence_logits[id_0] for sentence_logits in last_logits.squeeze(1)]

        return predictions

    def _sentences_to_prompt(self, sentences:str)->list[str]:
        """
        Format a list of sentences into proper format for LLM inference
        """
        prompts = []
        for i in range(len(sentences)):
            self.prompt[1]["content"] = sentences[i]
            prompts.append(self.tokenizer.apply_chat_template(self.prompt, tokenize=False))

        return prompts