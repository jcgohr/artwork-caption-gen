import numpy as np
import json
import csv
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sentence_transformers import SentenceTransformer

class SimClassifier:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def label_search(self, test_dataset, sentence_embeddings, labels, top_n:int=1):
        input_embeddings, true_labels = self.compute_embeddings(test_dataset)
        cosine_similarities = self._cosine_similarity(input_embeddings, sentence_embeddings)
        top_indices = np.argsort(-cosine_similarities, axis=1)[:,:top_n]

        predicted_labels = []
        for row in top_indices:
            visual_labels = 0
            contextual_labels = 0
            for column in row:
                if labels[column] == 1:
                    visual_labels+=1
                else:
                    contextual_labels+=1
            predicted_labels.append(visual_labels>contextual_labels) # predicted label for input sentence, based on most similar sentence labels

        return true_labels, predicted_labels

    def compute_embeddings(self, dataset_path:str, save_embs_path:str=None)->tuple[list,list]:
        texts, labels = self._format_dataset_for_retrieval(dataset_path)
        embeddings = self.model.encode(texts)
        if save_embs_path:
            np.savez(save_embs_path, embeddings=embeddings, labels=labels)
        return embeddings, labels

    def load_embeddings(self, embeddings_path:str)->tuple[list,list]:
        data = np.load(embeddings_path, allow_pickle=True)
        return data["embeddings"], data["labels"]
    
    def eval(self, y_true, y_pred, output_folder_name, override_output:bool=False):
        output_path = os.path.join("results", "classification_experiment")
        full_output_path = os.path.join(output_path, output_folder_name)
        if not os.path.exists(full_output_path):
            os.mkdir(full_output_path)
        else:
            if not override_output:
                raise FileExistsError(f"Directory {full_output_path} already exists.")

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

    def _format_dataset_for_retrieval(self, dataset_path:str)->tuple[list,list]:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        texts, labels = [], []
        for data in dataset.values():
            labels += [1] * len(data["visual_sentences"]) + [0] * len(data["contextual_sentences"])
            for sentence in data["visual_sentences"] + data["contextual_sentences"]:
                texts.append(sentence)

        return texts, labels

    def _cosine_similarity(self, a, b):
        dot_product = np.dot(a, b.T)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        return dot_product / (a_norm * b_norm.T)

if __name__ == '__main__':
    classifier = SimClassifier()
    embs, labs = classifier.load_embeddings("src/embeddings/allmpnet_train_embs.npz")
    true, pred = classifier.label_search("/mnt/netstore1_home/aidan.bell@maine.edu/artpedia/artpedia_test.json", embs, labs, 1)
    classifier.eval(true, pred, "sim_classifier")

