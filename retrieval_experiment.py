from submodules.longclip import model as longclip
from src.finetuner import finetune

import json
import torch
from PIL import Image
from ranx import Qrels, Run, evaluate

# def get_trained(val_split_path:str, train_split_path:str, caption_field:str, output_path:str, checkpoint_input_path: str = 'submodules/longclip/checkpoints/'):
#     """
#     Obtain finetuned longclip checkpoints from input data

#     Args:
#         val_split_path: Path to validation split json file
#         train_split_path: Path to train split json file
#         captions_path: Path to captions json file (generated from src/generate_captions.py)
#         caption_field: Name of val_split/train_split field in which captions are stored
#         output_path: Location to save checkpoints
#         checkpoint_input_path: Path of checkpoints to be finetuned
#     """
    
#     finetune(val_split_path, train_split_path, caption_field, output_path, checkpoint_input_path)
    
def search(checkpoint_path:str, test_split_path:dict)->Run:
    """
    Use longclip checkpoint for text to image retrieval

    Args:
        checkpoint_path: Path to longclip checkpoint file
        test_split_path: Path to test split json file

    Returns:
        Ranx run
        Ranx qrel
    """
    true_captions = _load_test(test_split_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load(checkpoint_path, device=device)

    caption_embeddings = torch.empty(len(true_captions), 768, device=device)
    image_embeddings = torch.empty(len(true_captions), 768, device=device)
    with torch.no_grad():
        for idx, sample in enumerate(true_captions.values()):
            caption_embeddings[idx, :] = model.encode_text(longclip.tokenize(sample["caption"], truncate=True).to(device))
            image_embeddings[idx, :] = model.encode_image(preprocess(Image.open(sample["file_path"])).unsqueeze(0).to(device))
    logits_per_caption = caption_embeddings @ image_embeddings.T

    # construct run/qrel
    caption_ids = list(true_captions.keys()) # for index to id conversion
    qrel_dict = {}
    run_dict = {}
    for idx, (id, sample) in enumerate(true_captions.items()):
        top_matching_indices = logits_per_caption[idx, :].argsort(dim=0, descending=True)[:100]
        values = logits_per_caption[idx, :][top_matching_indices]
        run_dict[id] = {}
        qrel_dict[id] = {id: 1}
        for key_idx, value in zip(top_matching_indices, values):
            run_dict[id][caption_ids[key_idx]] = value.item()
    return Qrels(qrel_dict, "t2i_retrieval"), Run(run_dict, "t2i_retrieval")

def eval(run_path:str, qrel_path:str, output_path:str)->None:
    with open(run_path, "r", encoding='utf-8') as f:
        run = json.load(f)
    with open(qrel_path, "r", encoding='utf-8') as f2:
        qrel = json.load(f2)
    results = evaluate(qrel, run, ["recall@1", "mrr"])
    with open(output_path, "w", encoding='utf-8') as out_f:
        json.dump(results, out_f, indent=4)

def _load_test(test_split_path:str)->dict:
    """
    Load test split into retrieval ready format

    Args:
        test_split_path: path to test split json file

    Returns:
        dict containing true captions and image paths
    """
    true_captions = {}
    with open(test_split_path, "r", encoding="utf-8") as f:
        test_split = json.load(f)
    for id, sample in test_split.items():
        true_captions[id] = {"caption": " ".join(sample["visual_sentences"]), "file_path": sample["file_path"]}
    return true_captions

if __name__ == '__main__':

    # qrel, run = search("submodules/longclip/checkpoints/longclip-L.pt", "/mnt/netstore1_home/aidan.bell@maine.edu/artpedia/artpedia_test.json")
    # run.save("results/retrieval_experiment/test.json")
    # qrel.save("results/retrieval_experiment/test_qrel.json")
    # results = evaluate(qrel, run, ["r@1, mrr"])
    # with open("results/retrieval_experiment/results.json", "w") as f:
    #     json.dump(results, f, indent=4)
    eval("results/retrieval_experiment/test.json", "results/retrieval_experiment/test_qrel.json", "results/retrieval_experiment/results.json")
    #finetune(val_split_path, train_split_path, caption_field, output_path, checkpoint_input_path)
