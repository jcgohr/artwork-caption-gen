from src.parsers import BlipCompleteParser
from src.utils import finetune_dataset_format
import retrieval_experiment as re

import os
import yaml
import subprocess

#TODO Make artpedia paths dynamic to user.
def complete_pipe(args):
    output_path = args.output_path
    if os.path.exists(output_path):
        raise FileExistsError(f"Output folder '{output_path}' already exists. Please delete it or provide a different folder name.")

    # create results directories
    results_path = os.path.join(output_path, "results")
    results_path_artpediaQ = os.path.join(results_path, "Artpedia_queries")
    results_path_llamaQ = os.path.join(results_path, "Llama_queries")
    os.makedirs(results_path_artpediaQ, exist_ok=False)
    os.makedirs(results_path_llamaQ, exist_ok=False)

    llava_cfg_path = os.path.join("submodules", "BLIP", "configs", "artpedia_llava_config.yaml")
    true_cfg_path = os.path.join("submodules", "BLIP", "configs", "artpedia_true_config.yaml")

    # update configs to have correct dataset paths
    with open(llava_cfg_path, 'r') as f:
        llava_cfg = yaml.safe_load(f)
    with open(true_cfg_path, 'r') as f:
        true_cfg = yaml.safe_load(f)

    # cfg keys and their updated values
    cfg_updates = {
        'train_ann_path': os.path.join(args.artpedia_path, 'artpedia_train.json'),
        'train_captions_path': os.path.join('captions', 'train_captions.json'),
        'val_ann_path': os.path.join(args.artpedia_path, 'artpedia_val.json'),
        'val_captions_path': os.path.join('captions', 'val_captions.json'),
        'test_ann_path': os.path.join(args.artpedia_path, 'artpedia_test.json'),
        'test_captions_path': os.path.join('captions', 'test_captions.json'),
        'batch_size_train': args.batch_size
    }

    # update cfg values
    for k,v in cfg_updates.items():
        llava_cfg[k] = v
        true_cfg[k] = v

    # overwrite old config
    with open(llava_cfg_path, 'w') as f:
        yaml.dump(llava_cfg, f)
    with open(true_cfg_path, 'w') as f:
        yaml.dump(true_cfg, f)

    # create runs
    ft_out_llava = os.path.join(output_path, "llava-caption-ft", "ft-files")
    ft_out_true = os.path.join(output_path, "human-caption-ft", "ft-files")
    runs = [
        (llava_cfg_path, 
         ft_out_llava), 
        (true_cfg_path,
         ft_out_true)
    ]

    # finetune
    for config_file, output_dir in runs:
        cmd = [
            "python", "-m", "torch.distributed.run",
            f"--nproc_per_node={args.gpus}",
            os.path.join('submodules', 'BLIP', 'train_retrieval.py'),
            "--config", config_file,
            "--output_dir", output_dir
        ]

        print(f"Finetuning using config@ -> {config_file}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("Subprocess failed.")
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            raise

    # search and eval
    baseline_checkpoint = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth"
    test_split = os.path.join(args.artpedia_path, "artpedia_test.json")
    llama_queries = os.path.join("captions", "generated_queries.json")

    qrel = re.build_qrel(test_split)
    baseline_run_artpedia_queries = re.blip_search(baseline_checkpoint, test_split)
    human_run_artpedia_queries = re.blip_search(os.path.join(ft_out_true, 'checkpoint_best.pth'), test_split)
    llava_run_artpedia_queries = re.blip_search(os.path.join(ft_out_llava, 'checkpoint_best.pth'), test_split)
    baseline_run_llama_queries = re.blip_search(baseline_checkpoint, test_split, generated_queries_path=llama_queries)
    human_run_llama_queries = re.blip_search(os.path.join(ft_out_true, 'checkpoint_best.pth'), test_split, generated_queries_path=llama_queries)
    llava_run_llama_queries = re.blip_search(os.path.join(ft_out_llava, 'checkpoint_best.pth'), test_split, generated_queries_path=llama_queries)
    re.eval(baseline_run_artpedia_queries, qrel, os.path.join(results_path_artpediaQ, "baseline.json"), ["precision@1", "mrr"], return_mean=True)
    re.eval(human_run_artpedia_queries, qrel, os.path.join(results_path_artpediaQ, "human-ft.json"), ["precision@1", "mrr"], return_mean=True)
    re.eval(llava_run_artpedia_queries, qrel, os.path.join(results_path_artpediaQ, "llava-ft.json"), ["precision@1", "mrr"], return_mean=True)
    re.eval(baseline_run_llama_queries, qrel, os.path.join(results_path_llamaQ, "baseline.json"), ["precision@1", "mrr"], return_mean=True)
    re.eval(human_run_llama_queries, qrel, os.path.join(results_path_llamaQ, "human-ft.json"), ["precision@1", "mrr"], return_mean=True)
    re.eval(llava_run_llama_queries, qrel, os.path.join(results_path_llamaQ, "llava-ft.json"), ["precision@1", "mrr"], return_mean=True)

if __name__ == "__main__":
    parser = BlipCompleteParser()
    args = parser.parse_args()
    complete_pipe(args)