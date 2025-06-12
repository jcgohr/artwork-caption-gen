import argparse

class BlipCompleteParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Perform full experiment with BLIP")
        self._add_arguments()

    def parse_args(self):
        return self.parser.parse_args()

    def _add_arguments(self):
        self.parser.add_argument("--artpedia_path", "-p", type=str, required=True, help="Path to artpedia dataset folder")
        self.parser.add_argument("--output_path", "-o", type=str, required=True, help="Folder path you want to store the experiment in")
        self.parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size for training")
        self.parser.add_argument("--gpus", "-g", type=str, default=1, help="Number of GPUs to run finetuning on")