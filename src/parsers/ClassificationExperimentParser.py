import argparse
import json

class ClassificationExperimentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Run LLM-based and optionally baseline classifiers on a dataset.")
        self._add_arguments()
        
    def _add_arguments(self):
        """Add the required and optional arguments to the parser."""
        self.parser.add_argument("dataset_path", type=str, help="Path to the dataset to classify.")
        self.parser.add_argument("prompt_path", type=str,  help="Path to the prompt to use for LLM classification.")
        self.parser.add_argument("output_folder_name", type=str, help="Desired output folder name.")
        self.parser.add_argument("afs_dataset_path", type=str, nargs="?", default=None, help="Path to the auto few-shot dataset (optional).")
        self.parser.add_argument("afs_top_n", type=int, nargs="?", default=None, help="Number of example sentences to use for AFS.")
        self.parser.add_argument("--run_baseline", action="store_true", help="Run baseline classifier along with LLM.")
        self.parser.add_argument("--overwrite_output", action="store_true", help="If output_folder_name already exists, override the files in it.")
    
    def _validate_json_file(self, file_path: str) -> bool:
        """
        Validate if the given file is a valid JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            with open(file_path, 'r',encoding='utf-8') as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False
            
    def parse_args(self) -> argparse.Namespace:
        """
        Parse and validate the command line arguments.
        
        Returns:
            argparse.Namespace: Parsed arguments
            
        Raises:
            ValueError: If the input paths are invalid
        """
        args = self.parser.parse_args()

        if (args.afs_dataset_path is None) != (args.afs_top_n is None):
            raise ValueError("Both 'afs_dataset_path' and 'afs_top_n' must be provided together or omitted.")
        
        # Validate metadata files
        if not self._validate_json_file(args.dataset_path):
            raise ValueError(f"Input metadata file '{args.dataset_path}' does not exist or is not a valid JSON file")
        if args.afs_dataset_path and not self._validate_json_file(args.afs_dataset_path):
            raise ValueError(f"Input metadata file '{args.afs_dataset_path}' does not exist or is not a valid JSON file")
            
        return args