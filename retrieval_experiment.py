from submodules.Long_CLIP import model as longclip
from submodules.BLIP.models.blip_itm import blip_itm
from src.parsers.RetrievalExperimentParser import RetrievalExperimentParser

import json
import torch
import csv
import os
from tqdm import tqdm
from PIL import Image
from ranx import Qrels, Run, evaluate
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def build_qrel(test_split_path:str, output_path:str=None)->Qrels:
    true_captions = _load_test(test_split_path)
    qrel_dict = {}
    for id in true_captions.keys():
        qrel_dict[id] = {id: 1}
    qrel = Qrels(qrel_dict, "t2i_retrieval")
    if output_path:
        qrel.save(output_path)
    return qrel
    
def longclip_search(checkpoint_path:str, test_split_path:str, output_path:str=None, generated_queries_path:str=None)->Run:
    """
    Use longclip checkpoint for text to image retrieval

    Args:
        checkpoint_path: Path to longclip checkpoint file
        test_split_path: Path to test split json file
        output_path (optional): Save run to this path

    Returns:
        Ranx run
    """
    true_captions = _load_test(test_split_path, generated_queries_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load(checkpoint_path, device=device)
    model.eval()

    caption_embeddings = torch.empty(len(true_captions), 768, device=device)
    image_embeddings = torch.empty(len(true_captions), 768, device=device)
    with torch.no_grad():
        for idx, sample in tqdm(enumerate(true_captions.values()), total=len(true_captions), desc="Performing Retrieval"):
            caption_embeddings[idx, :] = model.encode_text(longclip.tokenize(sample["caption"], truncate=True).to(device))
            image_embeddings[idx, :] = model.encode_image(preprocess(Image.open(sample["file_path"])).unsqueeze(0).to(device))
    logits_per_caption = caption_embeddings @ image_embeddings.T

    return construct_run(true_captions, logits_per_caption, "LongCLIP_retrieval", output_path)

def blip_search(checkpoint_path:str, test_split_path:str, output_path:str=None, generated_queries_path:str=None)->Run:
    """
    Use BLIP checkpoint for text to image retrieval

    Args:
        checkpoint_path: Path to longclip checkpoint file
        test_split_path: Path to test split json file
        output_path (optional): Save run to this path

    Returns:
        Ranx run
    """
    true_captions = _load_test(test_split_path, generated_queries_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = blip_itm(pretrained=checkpoint_path, image_size=384, vit="large" if "large" in os.path.basename(checkpoint_path) else "base",itc_sim=False)
    model.eval()
    model = model.to(device=device)
    
    caption_embeddings = torch.empty(len(true_captions), 768, device=device)
    image_embeddings = torch.empty(len(true_captions), 768, device=device)
    with torch.no_grad():
        for idx, sample in tqdm(enumerate(true_captions.values()), total=len(true_captions), desc="Performing Retrieval"):
            text_feat,image_feat = model(_load_image_for_BLIP(sample["file_path"], device=device), sample["caption"], match_head='itc')
            caption_embeddings[idx, :] = text_feat
            image_embeddings[idx, :] = image_feat
    logits_per_caption = caption_embeddings @ image_embeddings.T
                
    return construct_run(true_captions, logits_per_caption, "BLIP_retrieval", output_path)

def construct_run(true_captions:list, logits_per_caption:list, run_name:str, output_path:str=None, top_n:int=100):
    caption_ids = list(true_captions.keys()) # for index to id conversion
    run_dict = {}
    for idx, (id, _) in enumerate(true_captions.items()):
        top_matching_indices = logits_per_caption[idx, :].argsort(dim=0, descending=True)[:top_n]
        values = logits_per_caption[idx, :][top_matching_indices]
        run_dict[id] = {}
        for key_idx, value in zip(top_matching_indices, values):
            run_dict[id][caption_ids[key_idx]] = value.item()
    run = Run(run_dict, run_name)
    if output_path:
        run.save(output_path)
    return run

def eval(run:Run, qrel:str, output_path:str, metrics:list, return_mean:bool)->None:
    """
    If return_mean == True, output_path must be a .json. If false, output_path must be a .csv
    """
    results = evaluate(qrel, run, metrics, return_mean=return_mean)
    if not return_mean:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric"] + list(run.keys()))
            for metric in metrics:
                writer.writerow([metric] + results[metric].tolist())
    else:
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=4)


def _load_test(test_split_path:str, generated_queries_path:str=None)->dict:
    """
    Load test split into retrieval ready format

    Args:
        test_split_path: path to test split json file
        generated_queries_path: if you are running retrieval on queries generated by a captioner, then pass in the path to those queries
    Returns:
        dict containing true captions and image paths
    """
    true_captions = {}
    with open(test_split_path, "r", encoding="utf-8") as f:
        test_split = json.load(f)
    
    if generated_queries_path:
        with open(generated_queries_path, "r", encoding='utf-8') as f2:
            queries = json.load(f2)
    else:
        queries = test_split

    for id, sample in queries.items():
        if generated_queries_path:
            true_captions[id] = {"caption": next(iter(sample.values())), "file_path": test_split[id]["file_path"]}
        else: 
            true_captions[id] = {"caption": " ".join(sample["visual_sentences"]), "file_path": sample["file_path"]}
    return true_captions

def _load_image_for_BLIP(img_path, device, image_size=384): 
    raw_image = Image.open(img_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

if __name__ == '__main__':

    parser = RetrievalExperimentParser()
    args = parser.parse_args()

    if args.save_qrel:
        qrel = build_qrel(args.test_split_path, args.qrel_path)
    else:
        with open(args.qrel_path, "r", encoding="utf-8") as f:
            qrel = json.load(f)
    
    if args.using == "longclip":
        run = longclip_search(args.checkpoint_path, args.test_split_path, args.save_run, args.generated_queries)
    elif args.using == "blip":
        run = blip_search(args.checkpoint_path, args.test_split_path, args.save_run, args.generated_queries)
    eval(run, qrel, args.results_path, ["precision@1", "mrr"], return_mean=not args.eval_queries)
