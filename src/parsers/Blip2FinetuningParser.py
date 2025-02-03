import argparse
import os
from pathlib import Path

class BlipArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='BLIP Training Arguments')
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument(
            '--train-file',
            type=self._validate_json_file,
            required=True,
            help='Path to training data JSON file'
        )
        self.parser.add_argument(
            '--val-file',
            type=self._validate_json_file,
            required=True,
            help='Path to validation data JSON file'
        )
        self.parser.add_argument(
            '--train-captions-file',
            type=self._validate_json_file,
            required=True,
            help='Path to training captions JSON file'
        )
        self.parser.add_argument(
            '--val-captions-file',
            type=self._validate_json_file,
            required=True,
            help='Path to validation captions JSON file'
        )
        self.parser.add_argument(
            '--caption_key',
            type=str,
            required=True,
            help='Which captions to use e.g true, LlamaCaptioner, LlavaCaptioner'
        )
        self.parser.add_argument(
            '--output-dir',
            type=self._handle_output_dir,
            required=True,
            help='Directory to save output files'
        )

    def _validate_json_file(self, file_path: str) -> str:
        if not file_path.endswith('.json'):
            self.parser.error(f'File {file_path} must have a .json extension')
        
        if not os.path.exists(file_path):
            self.parser.error(f'File {file_path} does not exist')
            
        return file_path

    def _handle_output_dir(self, dir_path: str) -> str:
        output_path = Path(dir_path)
        output_path.mkdir(parents=True, exist_ok=True)
        return str(output_path)

    def parse_args(self):
        return self.parser.parse_args()