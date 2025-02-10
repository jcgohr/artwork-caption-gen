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