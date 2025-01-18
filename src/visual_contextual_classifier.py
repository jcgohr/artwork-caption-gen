import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
DEFAULT_PROMPT = [
    {"role": "system",
        "content": (
        "You are a painting description classifier that outputs ONLY the number '1' or '0'. Output '1' if the description the user gives you contains primarily visual details about the painting. If it doesn't, output '0'.")},

    {"role": "user", 
        "content": "<insert sentence>"}
]
# EXTRA_INSTRUCT = ("Sentence: {sentence}"
#                 "Output '1' if the sentence describes visual (imagery) descriptions."
#                 "Output '0' if the sentence gives contextual (history) descriptions.")

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

    def classify(self, sentences:list[str])->list[bool]:
        """
        Classify a list of sentences as either 1 (visual), or 0 (contextual)
        """
        formatted_prompts = self._sentences_to_prompt(sentences)
        skip_token = self.tokenizer.bos_token
        inputs = self.tokenizer(formatted_prompts, padding=True, padding_side="left", return_tensors="pt").to(self.model.device)

        final_logits = [] # Will contain the final logits tensor for each sentence
        while(True):
            with torch.no_grad():
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_next_token_ids = torch.argmax(logits[:, -1, :], dim=-1).tolist()
                predicted_next_token_vals = self.tokenizer.convert_ids_to_tokens(predicted_next_token_ids)
                
                # If any sentences in the batch generate the skip_token, then continued generation is needed
                needs_generation = []
                for idx, value in enumerate(predicted_next_token_vals):
                    if(value == skip_token):
                        needs_generation.append(idx)
                    else:
                        final_logits.append(logits[idx,-1,:])

                if not needs_generation:
                    break
                        
                # Continue generating sequences if needed
                for sentence_num in needs_generation:
                    # Add generated token to end of input
                    inputs["input_ids"][sentence_num] = torch.cat(
                        [inputs["input_ids"][sentence_num], torch.tensor([predicted_next_token_ids[sentence_num]], device=inputs["input_ids"].device)]
                    )
                    # Update attention mask for generated token
                    inputs["attention_mask"][sentence_num] = torch.cat(
                        [inputs["attention_mask"][sentence_num], torch.tensor([1], device=inputs["attention_mask"].device)]
                    )

        # Get classifications from logits
        id_1 = self.tokenizer.convert_tokens_to_ids("1")
        id_0 = self.tokenizer.convert_tokens_to_ids("0")
        predictions = [(logits[id_1] > logits[id_0]).item() for logits in final_logits]

        return predictions

    def _sentences_to_prompt(self, sentences:str, extra_instruct:str=None)->list[str]:
        """
        Format a list of sentences into proper format for LLM inference
        """
        prompts = []
        for i in range(len(sentences)):
            if extra_instruct:
                self.prompt[1]["content"] = extra_instruct.format(sentence=sentences[i])
            else:
                self.prompt[1]["content"] = sentences[i]
            prompts.append(self.tokenizer.apply_chat_template(self.prompt, add_generation_prompt=True, tokenize=False))

        return prompts