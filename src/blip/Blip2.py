import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForImageTextRetrieval
from transformers import TrainingArguments, Trainer
from src.utils.mutate import finetune_dataset_format
import json
from PIL import Image
import os
from glob import glob

class CustomImageTextRetrievalDataset(Dataset):
    def __init__(self,data_file,captions_file,key,processor):
        self.processor = processor
        self.samples = []
        
        data=finetune_dataset_format(data_file,captions_file)
        # Add each caption with its corresponding image path
        for _, item in data.items():
            self.samples.append({
                'image_path': item["file_path"],
                'caption': item[key]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        caption = sample['caption']

        # Process the image and text
        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding='max_length',
            truncation=True
        )

        # Remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        # Add labels for contrastive learning
        # In BLIP image-text retrieval, we use 1 for matched pairs
        encoding['labels'] = torch.tensor(1)

        return encoding

def collate_fn(batch):
    """
    Collate function to handle batching of processed examples
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }

def train_blip(
    train_file:str,
    val_file:str,
    train_captions_file:str,
    val_captions_file:str,
    output_dir,
    model_name:str="Salesforce/blip-image-text-retrieval",
    batch_size:int=16,
    num_epochs:int=3,
    learning_rate:float=2e-5
):
    # Load model and processor
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForImageTextRetrieval.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )

    # Create datasets
    train_dataset = CustomImageTextRetrievalDataset(train_file, processor)
    val_dataset = CustomImageTextRetrievalDataset(val_file, processor)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.05,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=4
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model()
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    # Example usage
    train_blip(
        train_dir="data/train",
        val_dir="data/val",
        output_dir="./blip_finetuned"
    )