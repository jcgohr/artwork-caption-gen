import argparse

class LongCompleteParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Perform full experiment with LongCLIP")
        self._add_arguments()

    def parse_args(self):
        return self.parser.parse_args()

    def _add_arguments(self):
        self.parser.add_argument("--output_path", "-o", type=str, required=True, help="Folder path you want to store the experiment in")
        self.parser.add_argument("--artpedia_path", "-p", type=str, required=True, help="Path to artpedia dataset folder")
        self.parser.add_argument("--checkpoint_in", "-ci", type=str, required=True, help="Path to LongCLIP .pt checkpoint you wish to finetune")
        self.parser.add_argument("--epochs", "-e", default=6, type=int, help="How many epochs to finetune over")
        self.parser.add_argument("--batch_size", "-b", default=40, type=int, help="Batch size for finetuning. Bigger is better, but also more memory intensive.")