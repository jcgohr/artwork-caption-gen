import sys
import os
sys.path.append(os.getcwd())
from src.utils.auto_fewshot import AutoFewShot

import torch
import copy
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
# DEFAULT_PROMPT = [
#     {"role": "system",
#         "content": (
#         "You are a painting caption classifier that outputs ONLY the number '1' or '0'. " 
#         "Output '1' if the caption the user gives you contains mostly details of visual contents within the painting. "
#         "Output '0' if the caption the user gives you contains mostly details of context that isn't describing visual details in the painting.")},

#     {"role": "user", 
#         "content": (
#         "She's sitting on a seat covered with a rich drape acting as carpet too, and she has on her knees an open book, symbol of the happening Scriptures.")},

#     {"role": "assistant",
#         "content": (
#         "1")},

#     {"role": "user", 
#         "content": (
#         "Nor did it emphasise the significance of the manuscript or the rough manner in which Christ seems to energetically leaf through it, his play watched on by a near-indulgent Mary.")},

#     {"role": "assistant",
#         "content": (
#         "0")},

#     {"role": "user", 
#         "content": "<insert sentence>"}
# ]
# EXTRA_INSTRUCT = ("Sentence: {sentence}"
#                 "Output '1' if the sentence describes visual (imagery) descriptions."
#                 "Output '0' if the sentence gives contextual (history) descriptions.")

class Classifier:
    """
    Classify sentences as visual or contextual with an LLM
    """
    def __init__(self, model_id:str, prompt:str=None, prompt_path:str=None, afs_dataset_path:str=None, **kwargs):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=ACCESS_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=ACCESS_TOKEN,**kwargs)

        self.afs = None
        # load data for auto-few-shot
        if afs_dataset_path is not None:
            with open(afs_dataset_path, 'r', encoding='utf-8') as f:
                afs_raw_data = json.load(f)
                afs_data = []
                for entry in afs_raw_data.values():
                    sen_count = 0
                    for sentence in entry["visual_sentences"] + entry["contextual_sentences"]:
                        label = 1 if sen_count<len(entry["visual_sentences"]) else 0
                        afs_data.append((sentence, label))
                        sen_count+=1
                self.afs = AutoFewShot(afs_data)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if prompt is None and prompt_path is None:
            raise TypeError("Either prompt or prompt_path must be provided")
        elif prompt_path is not None:
            with open(prompt_path, 'r', encoding='utf-8') as f2:
                self.prompt = json.load(f2)
        else:
            self.prompt = prompt

    def classify(self, sentences:list[str], auto_fs:bool=False, afs_top_n:int=1)->list[bool]:
        """
        Classify a list of sentences as either 1 (visual), or 0 (contextual)

        Args:
            sentences: List of sentences to classify
            auto_fs: True will use auto-few-shot prompting
            afs_top_n: Number of example sentences to use with auto_fs
        """
        formatted_prompts = self._sentences_to_prompt(sentences, auto_fs, afs_top_n)
        # skip_token = self.tokenizer.bos_token
        inputs = self.tokenizer(formatted_prompts, padding=True, padding_side="left", return_tensors="pt").to(self.model.device)

        final_logits = [] # Will contain the final logits tensor for each sentence
        while(True):
            with torch.no_grad():
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits
                for sentence_logits in logits:
                    final_logits.append(sentence_logits[-1, :])
            break
                
                # predicted_next_token_ids = torch.argmax(logits[:, -1, :], dim=-1).tolist()
                # predicted_next_token_vals = self.tokenizer.convert_ids_to_tokens(predicted_next_token_ids)
                # # If any sentences in the batch generate the skip_token, then continued generation is needed
                # needs_generation = []
                # for idx, value in enumerate(predicted_next_token_vals):
                #     if(value == skip_token):
                #         needs_generation.append(idx)
                #     else:
                #         final_logits.append(logits[idx,-1,:])

                # if not needs_generation:
                #     break
                        
                # # Continue generating sequences if needed
                # for sentence_num in needs_generation:
                #     # Add generated token to end of input
                #     inputs["input_ids"][sentence_num] = torch.cat(
                #         [inputs["input_ids"][sentence_num], torch.tensor([predicted_next_token_ids[sentence_num]], device=inputs["input_ids"].device)]
                #     )
                #     # Update attention mask for generated token
                #     inputs["attention_mask"][sentence_num] = torch.cat(
                #         [inputs["attention_mask"][sentence_num], torch.tensor([1], device=inputs["attention_mask"].device)]
                #     )

        # Get classifications from logits
        id_1 = self.tokenizer.convert_tokens_to_ids("1")
        id_0 = self.tokenizer.convert_tokens_to_ids("0")
        predictions = [(logits[id_1] > logits[id_0]).item() for logits in final_logits]

        return predictions
    
    def _create_chat_entry(self, role, content=None):
        return {"role": role, "content": content}

    def _sentences_to_prompt(self, sentences:str, auto_fs:bool, afs_top_n:int, extra_instruct:str=None)->list[str]:
        """
        Format a list of sentences into proper format for LLM inference
        """
        prompts = []
        for i in range(len(sentences)):
            curr_prompt = copy.deepcopy(self.prompt)
            if extra_instruct:
                curr_prompt["content"] = extra_instruct.format(sentence=sentences[i])
            elif auto_fs:
                most_similar = self.afs.most_similar(sentences[i], afs_top_n)
                for item in most_similar:
                    curr_prompt.append(self._create_chat_entry("user", item[0]))
                    curr_prompt.append(self._create_chat_entry("assistant", item[1]))
                curr_prompt.append(self._create_chat_entry("user", sentences[i]))
            else:
                curr_prompt[-1]["content"] = sentences[i]
            prompts.append(self.tokenizer.apply_chat_template(curr_prompt, add_generation_prompt=True, tokenize=False))

        return prompts