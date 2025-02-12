def parse_metric_results(scores, caption_keys, results, metric_keys):
    # For ROUGE metric (nested list structure)
    if isinstance(metric_keys[0], list):
        # ROUGE returns list of dictionaries for each caption pair
        for caption_idx, caption_key in enumerate(caption_keys):
            # results[caption_idx] contains scores for one caption pair
            rouge_scores = results[caption_idx]
            # Iterate through each ROUGE type (1, 2, L)
            for metric_group_idx, metric_group in enumerate(metric_keys):
                # Get p, r, f1 values for this ROUGE type
                for value_idx, metric_key in enumerate(metric_group):
                    scores[caption_key][metric_key] = rouge_scores[metric_group_idx][value_idx]
    
    # For BERTScore (returns three separate lists)
    elif len(metric_keys) == 3 and all(k.endswith(('-p', '-r', '-f1')) for k in metric_keys):
        p_scores, r_scores, f1_scores = results
        for caption_idx, caption_key in enumerate(caption_keys):
            scores[caption_key][metric_keys[0]] = p_scores[caption_idx]
            scores[caption_key][metric_keys[1]] = r_scores[caption_idx]
            scores[caption_key][metric_keys[2]] = f1_scores[caption_idx]
    
    # For BLEU and METEOR (single score per caption)
    else:
        for caption_idx, caption_key in enumerate(caption_keys):
            scores[caption_key][metric_keys[0]] = results[caption_idx]
    
    return scores

def write_scores_to_tsv(scores_dict, output_path, captions, caption_key):
    """
    Write evaluation scores and corresponding captions to a TSV file.
    
    Args:
        scores_dict: Dictionary of metric scores for each example
        output_path: Path to write the TSV file
        captions: Dictionary containing captions for each example
        caption_key: Key to identify the generated caption in the captions dict
    """
    # Get all metric keys from the first entry
    first_key = next(iter(scores_dict))
    metric_keys = list(scores_dict[first_key].keys())
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Write header with additional caption columns
        header = ['id'] + ['True_caption', f'{caption_key}_caption'] + metric_keys
        f.write('\t'.join(header) + '\n')
        
        # Write data rows
        for id_key, metrics in scores_dict.items():
            # Get captions for this example and remove newlines
            true_caption = captions[id_key]["True"].replace('\n', ' ').replace('\r', ' ')
            generated_caption = captions[id_key][caption_key].replace('\n', ' ').replace('\r', ' ')
            
            # Combine all fields for the row
            row = [
                str(id_key),
                true_caption,
                generated_caption
            ] + [str(metrics[key]) for key in metric_keys]
            
            f.write('\t'.join(row) + '\n')