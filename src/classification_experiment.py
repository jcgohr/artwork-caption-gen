from visual_contextual_classifier import classifier, torch

sentences = ["The sky is a brilliant blue and the sun shines brightly in the distance.", "The author tragically died in 1934 at 27 years of age."]
classifier_1b = classifier("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
# classifier_3b = classifier("meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

predictions = classifier_1b.classify(sentences)
for i in range(len(predictions)):
    print(sentences[i] + ": " + ("visual" if predictions[i].item() == 1 else "contextual"))