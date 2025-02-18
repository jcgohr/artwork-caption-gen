# Artwork Caption Generation

Jimmy Gore, Aidan Bell

## Description

This repo is used for analyzing the capabilities of Multimodal Large Language Models (MLLMs) as artwork captioners.

## Research questions

### Q1: Can MLLMs provide high quality captions for paintings?

### Q2: Can MLLM generated painting captions be effective for fine-tuning?

## Data

Artpedia

- 2930 Paintings, and their associated captions + metadata. Captions are organized into "visual" or "contextual" bins for each painting. https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=35

## Related Research

- Bucciarelli, D., Moratelli, N., Cornia, M., Baraldi, L., Cucchiara, R., et al.: Personalizing Multimodal Large Language Models for Image Captioning: An Experimental Analysis. In: ECCV Workshops (2024).

## Installation

### Requirements

## `retrieval_experiment.py`


### BLIP Retrieval With Llama Queries
```
python retrieval_experiment.py blip/llava_large/llava_large.pth blip artpedia/artpedia_test.json results/retrieval_experiment/blip_results/Lllama_queries/llava_mean.json results/retrieval_experiment/qrel.json --save_run results/retrieval_experiment/blip_results/Lllama_queries/llava_scores.json --save_qrel --generated_queries captions/generated_queries.json
```

```
python retrieval_experiment.py blip/true_large/true_large.pth blip artpedia/artpedia_test.json results/retrieval_experiment/blip_results/Lllama_queries/true_mean.json results/retrieval_experiment/qrel.json --save_run results/retrieval_experiment/blip_results/Lllama_queries/true_scores.json --save_qrel --generated_queries captions/generated_queries.json
```

```
python retrieval_experiment.py blip/baseline_large/baseline_large.pth blip artpedia/artpedia_test.json results/retrieval_experiment/blip_results/Lllama_queries/baseline_mean.json results/retrieval_experiment/qrel.json --save_run results/retrieval_experiment/blip_results/Lllama_queries/baseline_run.json --save_qrel --generated_queries captions/generated_queries.json
```