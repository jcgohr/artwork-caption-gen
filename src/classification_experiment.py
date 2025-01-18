from visual_contextual_classifier import classifier
import torch
import json
import sys
import csv
import os
import matplotlib.pyplot as plt
from VisualContextualClassifier import VisualContextualClassifier
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

RUN_BASELINE = False # True will also run baseline classifier, False is just LLM

dataset_path = sys.argv[1]
if not os.path.exists(dataset_path):
   raise OSError(f"Dataset path {dataset_path} does not exist")
classifier_8B = classifier("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
baseline_classifier = VisualContextualClassifier() if RUN_BASELINE else None

with open(dataset_path, "r", encoding='utf-8') as f:
    dataset = json.load(f)
dataset = [entry for entry in dataset.values() if entry.get('split') == 'test']

# generate true/pred
y_true = []
y_pred = []
baseline_y_pred = []
for data in tqdm(dataset, desc="Image captions classified"):
    predictions = classifier_8B.classify(data["visual_sentences"] + data["contextual_sentences"])
    y_true += [1] * len(data["visual_sentences"]) + [0] * len(data["contextual_sentences"])
    y_pred += predictions
    if RUN_BASELINE:
        for sentence in data["visual_sentences"] + data["contextual_sentences"]:
            b_pred = baseline_classifier.predict(sentence)
            baseline_y_pred.append(1 if max(b_pred, key=b_pred.get) == "visual" else 0)


output_path = os.path.join("results", "classification_experiment")

# eval
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

# baseline eval
if RUN_BASELINE:
    f1_b = f1_score(y_true, baseline_y_pred)
    accuracy_b = accuracy_score(y_true, baseline_y_pred)

    with open(os.path.join(output_path, "baseline_eval.csv"), "w", newline="") as f3:
        writer = csv.writer(f3)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", accuracy_b])
        writer.writerow(["F1 Score", f1_b])

    cm_b = confusion_matrix(y_true, baseline_y_pred)
    cm_display_b = ConfusionMatrixDisplay(confusion_matrix=cm_b, display_labels=["Contextual", "Visual"])
    cm_display_b.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(output_path, "baseline_confusion_matrix.png"), dpi=300, bbox_inches="tight")



