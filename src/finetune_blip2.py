from src.blip import train_blip

if __name__ == "__main__":
    # Example usage
    train_blip(
        train_dir="data/train",
        val_dir="data/val",
        output_dir="./blip_finetuned"
    )