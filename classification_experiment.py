from src.visual_contextual_classifier import Classifier
from src.parsers.ClassificationExperimentParser import ClassificationExperimentParser

import torch
import json
import csv
import os
import matplotlib.pyplot as plt
from VisualContextualClassifier import VisualContextualClassifier
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# argparsing
parser = ClassificationExperimentParser()
args = parser.parse_args()

classifier_8B = Classifier(
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
    predictions = classifier_8B.classify(data["visual_sentences"] + data["contextual_sentences"], auto_fs=args.afs_dataset_path is not None, afs_top_n=args.afs_top_n)
    y_true += [1] * len(data["visual_sentences"]) + [0] * len(data["contextual_sentences"])
    y_pred += predictions
    if baseline_classifier:
        for sentence in data["visual_sentences"] + data["contextual_sentences"]:
            b_pred = baseline_classifier.predict(sentence)
            baseline_y_pred.append(1 if max(b_pred, key=b_pred.get) == "visual" else 0)

output_path = os.path.join("results", "classification_experiment")
full_output_path = os.path.join(output_path, args.output_folder_name)
if not os.path.exists(full_output_path):
    os.mkdir(full_output_path)

# eval
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
with open(os.path.join(full_output_path, "eval.csv"), "w", newline="") as f2:
    writer = csv.writer(f2)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Accuracy", accuracy])
    writer.writerow(["F1 Score", f1])

cm = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Contextual", "Visual"])
cm_display.plot(cmap=plt.cm.Greens)
plt.savefig(os.path.join(full_output_path, "confusion_matrix.png"), dpi=300, bbox_inches="tight")

# baseline eval
if baseline_classifier:
    base_output_path = os.path.join(output_path, "baseline")
    if not os.path.exists(base_output_path):
        os.mkdir(base_output_path)
    f1_b = f1_score(y_true, baseline_y_pred)
    accuracy_b = accuracy_score(y_true, baseline_y_pred)

    with open(os.path.join(output_path, "eval.csv"), "w", newline="") as f3:
        writer = csv.writer(f3)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", accuracy_b])
        writer.writerow(["F1 Score", f1_b])

    cm_b = confusion_matrix(y_true, baseline_y_pred)
    cm_display_b = ConfusionMatrixDisplay(confusion_matrix=cm_b, display_labels=["Contextual", "Visual"])
    cm_display_b.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(base_output_path, "confusion_matrix.png"), dpi=300, bbox_inches="tight")



