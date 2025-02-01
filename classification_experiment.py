from src.visual_contextual_classifier import classifier

import argparse
import torch
import json
import sys
import csv
import os
import matplotlib.pyplot as plt
from VisualContextualClassifier import VisualContextualClassifier
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# argparsing
parser = argparse.ArgumentParser(description="Run LLM-based and optionally baseline classifiers on a dataset.")
parser.add_argument("dataset_path", type=str, help="Path to the dataset to classify.")
parser.add_argument("prompt_path", type=str,  help="Path to the prompt to use for LLM classification.")
parser.add_argument("afs_dataset_path", type=str, nargs="?", default=None, help="Path to the auto few-shot dataset (optional).")
parser.add_argument("--run_baseline", action="store_true", help="Run baseline classifier along with LLM.")
args = parser.parse_args()

if not os.path.exists(args.dataset_path):
    raise OSError(f"Dataset path {args.dataset_path} does not exist")

auto_fs = args.afs_dataset_path is not None
afs_top_n = 2 if auto_fs else None

if auto_fs and not os.path.exists(args.afs_dataset_path):
    raise OSError(f"Dataset path {args.afs_dataset_path} does not exist")

classifier_8B = classifier(
    "meta-llama/Llama-3.1-8B-Instruct", 
    prompt_path=args.prompt_path, 
    afs_dataset_path=args.afs_dataset_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
baseline_classifier = VisualContextualClassifier() if args.run_baseline else None
with open(args.dataset_path, "r", encoding='utf-8') as f:
    dataset = json.load(f)


# generate true/pred
y_true = []
y_pred = []
baseline_y_pred = []
for data in tqdm(dataset.values(), desc="Image captions classified"):
    predictions = classifier_8B.classify(data["visual_sentences"] + data["contextual_sentences"], auto_fs=auto_fs, afs_top_n=afs_top_n)
    y_true += [1] * len(data["visual_sentences"]) + [0] * len(data["contextual_sentences"])
    y_pred += predictions
    if baseline_classifier:
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
if baseline_classifier:
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



