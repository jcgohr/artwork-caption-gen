import os
import argparse
import json

class RetrievalExperimentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Test retrieval on model checkpoints")
        self._add_arguments()
        
    def _add_arguments(self):
        """Add the required and optional arguments to the parser."""
        self.parser.add_argument("checkpoint_path", type=str, help="Path to model checkpoints.")
        self.parser.add_argument("using", type=str, help="Which model to use for retrieval (longclip or blip).")
        self.parser.add_argument("test_split_path", type=str, help="Path to the test split.")
        self.parser.add_argument("results_path", type=str, help="Desired .json output filename to save results in.")
        self.parser.add_argument("qrel_path", type=str, help="Path to pre-built qrel .json file, or location to save qrel if using flag --save_qrel.")
        self.parser.add_argument("--save_qrel", action="store_true", help="Save qrel file into qrel_path (not needed if qrel_path is pre-made qrel.json file)")
        self.parser.add_argument("--save_run", type=str, nargs="?", default=None, metavar="run_path", help="Path to save run file (optional)")
    
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
        
        # Validate metadata files
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Input checkpoint path '{args.checkpoint_path}' does not exist")
        if not self._validate_json_file(args.test_split_path):
            raise ValueError(f"Input metadata file '{args.test_split_path}' does not exist or is not a valid JSON file")
        if not args.save_qrel and not self._validate_json_file(args.qrel_path):
            raise ValueError(f"Input qrel file '{args.qrel_path}' does not exist or is not a valid JSON file. Use --save_qrel to construct the qrel and save in qrel_path")
            
        return args