from src.parsers import BlipCompleteParser
from src.utils import finetune_dataset_format

import os
import yaml
import subprocess

#TODO Make artpedia paths dynamic to user.
def complete_pipe(args):
    output_path = args.output_path
    if os.path.exists(output_path):
        raise FileExistsError(f"Output folder '{output_path}' already exists. Please delete it or provide a different folder name.")
    # val_split = os.path.join(args.artpedia_path, "artpedia_val.json")
    # test_split = os.path.join(args.artpedia_path, "artpedia_test.json")
    # train_split = os.path.join(args.artpedia_path, "artpedia_train.json")
    # llama_queries = os.path.join("captions", "generated_queries.json")

    results_path = os.path.join(output_path, "results")
    results_path_artpediaQ = os.path.join(results_path, "Artpedia_queries")
    results_path_llamaQ = os.path.join(results_path, "Llama_queries")
    os.makedirs(results_path_artpediaQ, exist_ok=False)
    os.makedirs(results_path_llamaQ, exist_ok=True)

    # shape dataset as needed
    # val_data = finetune_dataset_format(val_split, os.path.join("captions", "val_captions.json"))
    # train_data = finetune_dataset_format(train_split, os.path.join("captions", "train_captions.json"))

    llava_cfg_path = os.path.join("submodules", "BLIP", "configs", "artpedia_llava_config.yaml")
    true_cfg_path = os.path.join("submodules", "BLIP", "configs", "artpedia_true_config.yaml")

    # update configs to have correct dataset paths
    with open(llava_cfg_path, 'r') as f:
        llava_cfg = yaml.safe_load(f)
    with open(true_cfg_path, 'r') as f:
        true_cfg = yaml.safe_load(f)

    # cfg keys and their updated subpaths
    path_keys = {
        'train_ann_path': 'artpedia_train.json',
        'train_captions_path': 'captions/train_captions.json',
        'val_ann_path': 'artpedia_val.json',
        'val_captions_path': 'captions/val_captions.json',
        'test_ann_path': 'artpedia_test.json',
        'test_captions_path': 'captions/test_captions.json'
    }

    # update paths
    for key, filename in path_keys.items():
        if 'captions' in key:
            new_path = filename
        else:
            new_path = os.path.join(args.artpedia_path, filename)
        llava_cfg[key] = str(new_path)
        true_cfg[key] = str(new_path)

    # overwrite old config
    with open(llava_cfg_path, 'w') as f:
        yaml.dump(llava_cfg, f)
    with open(true_cfg_path, 'w') as f:
        yaml.dump(true_cfg, f)

    # create runs
    runs = [
        (llava_cfg_path, 
         os.path.join(output_path, "llava-caption-ft", "ft-files")), 
        (true_cfg_path,
         os.path.join(output_path, "human-caption-ft", "ft-files"))
    ]

    # finetune
    for config_file, output_dir in runs:
        cmd = [
            "python", "-m", "torch.distributed.run",
            f"--nproc_per_node={args.gpus}",
            "submodules/BLIP/train_retrieval.py",
            "--config", config_file,
            "--output_dir", output_dir
        ]

        print(f"Finetuning using config@ -> {config_file}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("Subprocess failed.")
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            raise

    # search and eval

if __name__ == "__main__":
    parser = BlipCompleteParser()
    args = parser.parse_args()
    complete_pipe(args)