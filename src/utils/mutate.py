# stores functions that mutate data/datasets

import json
import os

def finetune_dataset_format(metadata_path:str, generated_cap_path:str, output_path:str|None):
    """
    A consistent dict format is required for finetuning, the following will be done:

    Merge true captions into generated captions under "True" field, and add file_path.
    The new dict will be written to output_path.

    Args:
        metadata_path: Path to metadata json file (artpedia)
        generated_cap_path: Path to generated captions file (obtained from generate_captions.py)
        output_path: Location to write new dict to
    """

    if output_path!=None and os.path.exists(output_path):
        raise FileExistsError

    with open(metadata_path, "r", encoding='utf-8') as f:
        metadata = json.load(f)
    with open(generated_cap_path, "r", encoding='utf-8') as f2:
        generated = json.load(f2)
    for id, sample in generated.items():
        sample["True"] = " ".join(metadata[id]["visual_sentences"])
        sample["file_path"] = metadata[id]["file_path"]

    if not output_path:
        return generated
    
    with open(output_path, "w", encoding='utf-8') as f3:
        f3.write(json.dumps(generated, indent=4, ensure_ascii=False))
                                   

