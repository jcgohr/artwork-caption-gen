# Artwork Caption Generation

Paper link to be provided.

## Description

This repo holds the code and experiments for our SIGIR 2025 short paper titled: "Augmenting Cross-Modal Art Retrieval: The Role of MLLM-Synthesized Captions". 
Our research explores the capabilities of Multimodal Large Language Models (MLLMs) as artwork captioners, with emphasis on the following two research questions.

## Research questions  

#### Q1: Is the quality of MLLM-generated artwork captions comparable to that of human-annotated captions? 

#### Q2: Are MLLM-generated artwork captions adequate for fine-tuning cross-modal retrieval models?  

## Paper TL;DR  
We compare LLaVA generated captions to ground-truth (human) captions of the [Artpedia](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=35) dataset. **Experiment #1:** We use several automatic text-similarity metrics (e.g. BERTScore, SPICE) and find that the semantic similarity between the synthetic/real captions is high, but lexical similarity is low. **Experiment #2:** We fine-tune two cross-modal retrieval models, BLIP and Long-CLIP, seperately on the generated and real captions. We compare the performance of the models when finetuned on sythetic versus real captions. We find that the models perform similarly with synthetic data, suggesting that MLLM-generated captions are sufficient for fine-tuning retrieval models.  

## Installation

**1.** Clone the repository:

   ```
   git clone --recursive https://github.com/jcgohr/artwork-caption-gen.git
   cd artwork-caption-gen
   ```

**2.** **TODO: Add remove corrupted script, add path modifier script...** ~~Download the [Artpedia dataset](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=35), and extract it.~~

**3.** Download the [Long-CLIP checkpoint](https://huggingface.co/BeichenZhang/LongCLIP-L) and place it in 'submodules/Long_CLIP/checkpoints/'

**4.** Install [Python 3.12](https://www.python.org/downloads/release/python-3120/)

**5.** Install the required dependencies:
```
pip install -r requirements.txt
```

## Run/Reproduce  
For ease of use, we provide a pipeline for each part of the experiment.  

**1. Automatic Text-Similarity Metrics**
```
TBD...
```
**2. Long-CLIP Fine-tuning + Evaluation**
```
Usage: python -m longclip_pipeline --output_path OUTPUT_PATH --artpedia_path ARTPEDIA_PATH --checkpoint_in CHECKPOINT_IN [--epochs EPOCHS] [--batch_size BATCH_SIZE]
```
```
Example: python -m longclip_pipeline --output_path longclip_experiment --artpedia_path storage/artpedia --checkpoint_in submodules/Long_CLIP/checkpoints/longclip-L.pt --epochs 4 --batch_size 30
```
**3. BLIP Fine-tuning + Evaluation**
```
Usage: python -m blip_pipeline --output_path OUTPUT_PATH --artpedia_path ARTPEDIA_PATH [--batch_size BATCH_SIZE] [--gpus GPUS]
```
```
Example: python -m blip_pipeline --output_path blip_experiment --artpedia_path storage/artpedia --batch_size 16
```
## Extra
**Note**  
The pipelines may be very slow depending on hardware. The BLIP fine-tuning especially requires a large amount of VRAM for large batch-sizes. Due to this, some of the hyper-parameters may not be exactly the same by default. Please refer to the paper for specific fine-tuning hyperparameters. 
