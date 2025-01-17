from visual_contextual_classifier import classifier, torch

sentences = ["The chimney breast was realized by the workshop of Niccolò dell' Abbate", "In this painting he depicts an old Dutch peasant taking his produce to the market, on his head a heavy tub, in his right hand a brace of mallard and a basket of eggs."]
classifier_8B = classifier("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

predictions = classifier_8B.classify(sentences)
for i in range(len(predictions)):
    print(sentences[i] + ": " + ("visual" if predictions[i].item() == 1 else "contextual"))