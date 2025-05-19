from src.finetuner import finetune 
from src.parsers import LongCompleteParser
from src.utils import finetune_dataset_format
import retrieval_experiment as re

import os
import gc
import torch

def complete_pipe(args):
    output_path = args.output_path
    if os.path.exists(output_path):
        raise FileExistsError(f"Output folder '{output_path}' already exists. Please delete it or provide a different folder name.")
    val_split = os.path.join(args.artpedia_path, "artpedia_val.json")
    test_split = os.path.join(args.artpedia_path, "artpedia_test.json")
    train_split = os.path.join(args.artpedia_path, "artpedia_train.json")
    llama_queries = os.path.join("captions", "generated_queries.json")

    results_path = os.path.join(output_path, "results")
    results_path_artpediaQ = os.path.join(results_path, "Artpedia_queries")
    results_path_llamaQ = os.path.join(results_path, "Llama_queries")
    os.makedirs(results_path_artpediaQ, exist_ok=False)
    os.makedirs(results_path_llamaQ, exist_ok=True)

    # shape dataset as needed
    val_data = finetune_dataset_format(val_split, os.path.join("captions", "val_captions.json"))
    train_data = finetune_dataset_format(train_split, os.path.join("captions", "train_captions.json"))

    # finetune
    ft1 = finetune(val_data, train_data, "True", checkpoint_output_path=os.path.join(output_path, "human-caption-ft"), epochs=args.epochs, batch_size=args.batch_size, early_stop=True, checkpoint_input_path=args.checkpoint_in)
    human_checkpoint = ft1.trainloop()
    del ft1
    gc.collect()
    torch.cuda.empty_cache()
    ft2 = finetune(val_data, train_data, "LlavaCaptioner", checkpoint_output_path=os.path.join(output_path, "llava-caption-ft"), epochs=args.epochs, batch_size=args.batch_size, early_stop=True, checkpoint_input_path=args.checkpoint_in)
    llava_checkpoint = ft2.trainloop()
    del ft2
    gc.collect()
    torch.cuda.empty_cache()

    # search and eval
    qrel = re.build_qrel(test_split)
    baseline_run_artpedia_queries = re.longclip_search(args.checkpoint_in, test_split)
    human_run_artpedia_queries = re.longclip_search(os.path.join(args.output_path, "human-caption-ft", "ft-checkpoints", human_checkpoint), test_split)
    llava_run_artpedia_queries = re.longclip_search(os.path.join(args.output_path, "llava-caption-ft", "ft-checkpoints", llava_checkpoint), test_split)
    baseline_run_llama_queries = re.longclip_search(args.checkpoint_in, test_split, generated_queries_path=llama_queries)
    human_run_llama_queries = re.longclip_search(os.path.join(args.output_path, "human-caption-ft", "ft-checkpoints", human_checkpoint), test_split, generated_queries_path=llama_queries)
    llava_run_llama_queries = re.longclip_search(os.path.join(args.output_path, "llava-caption-ft", "ft-checkpoints", llava_checkpoint), test_split, generated_queries_path=llama_queries)
    re.eval(baseline_run_artpedia_queries, qrel, os.path.join(results_path_artpediaQ, "baseline.json"), ["precision@1", "mrr"], return_mean=True)
    re.eval(human_run_artpedia_queries, qrel, os.path.join(results_path_artpediaQ, "human-ft.json"), ["precision@1", "mrr"], return_mean=True)
    re.eval(llava_run_artpedia_queries, qrel, os.path.join(results_path_artpediaQ, "llava-ft.json"), ["precision@1", "mrr"], return_mean=True)
    re.eval(baseline_run_llama_queries, qrel, os.path.join(results_path_llamaQ, "baseline.json"), ["precision@1", "mrr"], return_mean=True)
    re.eval(human_run_llama_queries, qrel, os.path.join(results_path_llamaQ, "human-ft.json"), ["precision@1", "mrr"], return_mean=True)
    re.eval(llava_run_llama_queries, qrel, os.path.join(results_path_llamaQ, "llava-ft.json"), ["precision@1", "mrr"], return_mean=True)

if __name__ == '__main__':
    parser = LongCompleteParser()
    args = parser.parse_args()
    complete_pipe(args)