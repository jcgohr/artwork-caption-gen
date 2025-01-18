from visual_contextual_classifier import classifier
import torch
import json
import sys
import csv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

dataset_path = sys.argv[1]
if not os.path.exists(dataset_path):
   raise OSError(f"Dataset path {dataset_path} does not exist")
classifier_8B = classifier("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

with open(dataset_path, "r", encoding='utf-8') as f:
    dataset = json.load(f)

# generate true/pred
y_true = []
y_pred = []
for id, data in tqdm(dataset.items(), desc="Image captions classified"):
    predictions = classifier_8B.classify(data["visual_sentences"] + data["contextual_sentences"])
    y_true += [1] * len(data["visual_sentences"]) + [0] * len(data["contextual_sentences"])
    y_pred += predictions

# eval
output_path = os.path.join("results", "classification_experiment")

f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
with open(os.path.join(output_path, "eval.csv"), "w", newline="") as f2:
    writer = csv.writer(f2)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Accuracy", accuracy])
    writer.writerow(["F1 Score", f1])

cm = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Contextual", "Visual"])
cm_display.plot(cmap=plt.cm.Greens)
plt.savefig(os.path.join(output_path, "confusion_matrix.png"), dpi=300, bbox_inches="tight")


