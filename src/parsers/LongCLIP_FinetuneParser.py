import argparse

class FinetuneParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Finetune LongCLIP")
        self._add_arguments()

    def parse_args(self):
        return self.parser.parse_args()

    def _add_arguments(self):
        self.parser.add_argument("--val_path", "-v", type=str, required=True, help="Path to validation split .json file")
        self.parser.add_argument("--train_path", "-t", type=str, required=True, help="Path to train split .json file")
        self.parser.add_argument("--cap", "-c", type=str, required=True, help="Name of the key with captions as values, within the val/train datasets")
        self.parser.add_argument("--checkpoint_in", "-ci", type=str, required=True, help="Path to LongCLIP .pt checkpoint you wish to finetune")
        self.parser.add_argument("--checkpoint_out", "-co", type=str, required=True, help="Path/folder name that you want to hold finetuned checkpoints + logs")
        self.parser.add_argument("--epochs", "-e", default=6, type=int, help="How many epochs to finetune over")
        self.parser.add_argument("--save_min", "-sm", action="store_true", help="Only save checkpoints if they have the current lowest validation loss")