from visual_contextual_classifier import classifier, torch

classifier_1b = classifier("meta-llama/Llama-3.2-1B-Instruct", kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"})
classifier_3b = classifier("meta-llama/Llama-3.2-3B-Instruct", kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"})