from submodules.longclip import model as longclip
from submodules.BLIP.models.blip_itm import blip_itm
from src.finetuner import finetune
from src.parsers.RetrievalExperimentParser import RetrievalExperimentParser

import json
import torch
import requests
from PIL import Image
from ranx import Qrels, Run, evaluate
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def build_qrel(test_split_path:str, output_path:str=None)->Qrels:
    true_captions = _load_test(test_split_path)
    qrel_dict = {}
    for id in true_captions.keys():
        qrel_dict[id] = {id: 1}
    qrel = Qrels(qrel_dict, "t2i_retrieval")
    if output_path:
        qrel.save(output_path)
    return qrel
    
def longclip_search(checkpoint_path:str, test_split_path:dict, output_path:str=None)->Run:
    """
    Use longclip checkpoint for text to image retrieval

    Args:
        checkpoint_path: Path to longclip checkpoint file
        test_split_path: Path to test split json file
        output_path (optional): Save run to this path

    Returns:
        Ranx run
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

    # construct run
    caption_ids = list(true_captions.keys()) # for index to id conversion
    run_dict = {}
    for idx, (id, sample) in enumerate(true_captions.items()):
        top_matching_indices = logits_per_caption[idx, :].argsort(dim=0, descending=True)[:100]
        values = logits_per_caption[idx, :][top_matching_indices]
        run_dict[id] = {}
        for key_idx, value in zip(top_matching_indices, values):
            run_dict[id][caption_ids[key_idx]] = value.item()
    run = Run(run_dict, "longclip_retrieval")
    if output_path:
        run.save(output_path)
    return run

def blip_search(checkpoint_path:str, test_split_path:dict, output_path:str=None)->Run:
    """
    Use BLIP checkpoint for text to image retrieval

    Args:
        checkpoint_path: Path to longclip checkpoint file
        test_split_path: Path to test split json file
        output_path (optional): Save run to this path

    Returns:
        Ranx run
    """
    true_captions = _load_test(test_split_path)

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = blip_itm(pretrained=checkpoint_path, image_size=384, vit='base')
    model.eval()
    model = model.to(device='cpu')
    for idx, sample in enumerate(true_captions.values()):
        logits_per_caption = model(_load_image_for_BLIP(sample["file_path"], device), true_captions["caption"], match_head='itc')

    # construct run
    caption_ids = list(true_captions.keys()) # for index to id conversion
    run_dict = {}
    for idx, (id, sample) in enumerate(true_captions.items()):
        top_matching_indices = logits_per_caption[idx, :].argsort(dim=0, descending=True)[:100]
        values = logits_per_caption[idx, :][top_matching_indices]
        run_dict[id] = {}
        for key_idx, value in zip(top_matching_indices, values):
            run_dict[id][caption_ids[key_idx]] = value.item()
    run = Run(run_dict, "BLIP_retrieval")
    if output_path:
        run.save(output_path)
    return run

def eval(run:str, qrel:str, output_path:str)->None:
    results = evaluate(qrel, run, ["recall@1", "mrr"])
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)

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

def _load_image_for_BLIP(img_url, device, image_size=384): 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   
    
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
        run = longclip_search(args.checkpoint_path, args.test_split_path, args.save_run)
    elif args.using == "blip":
        run = blip_search(args.checkpoint_path, args.test_split_path, args.save_run)
    eval(run, qrel, args.results_path)
