import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict

class CaptionGenerationParser:
    """
    Argument parser for caption generation tasks that handles input metadata,
    output file paths, and optional class filtering.
    """
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Process metadata and generate captions with optional class filtering'
        )
        self._add_arguments()
        
    def _add_arguments(self):
        """Add the required and optional arguments to the parser."""
        self.parser.add_argument(
            '-m', '--metadata',
            type=str,
            required=True,
            help='Path to the input JSON metadata file'
        )
        
        self.parser.add_argument(
            '-o', '--output',
            type=str,
            required=True,
            help='Path to save the output JSON file'
        )
        
        self.parser.add_argument(
            '-c', '--config',
            type=str,
            required=True,
            help='Path to JSON config file containing class parameters'
        )

        self.parser.add_argument(
            '-cl', '--classes',
            type=str,
            nargs='+',
            help='Optional list of class names to filter the metadata',
            default=None
        )
    
    def _validate_json_file(self, file_path: str) -> bool:
        """
        Validate if the given file is a valid JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False
            
    def parse_args(self) -> argparse.Namespace:
        """
        Parse and validate the command line arguments.
        Loads both metadata and config JSON files into dictionaries.
        
        Returns:
            argparse.Namespace: Parsed arguments with loaded dictionaries
            
        Raises:
            ValueError: If the input paths are invalid
        """
        args = self.parser.parse_args()
        
        # Validate and load metadata file
        if not self._validate_json_file(args.metadata):
            raise ValueError(
                f"Input metadata file '{args.metadata}' does not exist or is not a valid JSON file"
            )
        with open(args.metadata, 'r') as f:
            args.metadata = json.load(f)
            
        # Validate and load config file
        if not self._validate_json_file(args.config):
            raise ValueError(
                f"Config file '{args.config}' does not exist or is not a valid JSON file"
            )
        with open(args.config, 'r') as f:
            args.config = json.load(f)
        
        # Create output directory if it doesn't exist
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
            
        return args

# Example usage:
if __name__ == '__main__':
    parser = CaptionGenerationParser()
    try:
        args = parser.parse_args()
        print("Loaded metadata dictionary")
        if args.classes:
            print(f"Filtering for classes: {', '.join(args.classes)}")
        print("Config parameters:")
        for class_name, params in args.config.items():
            print(f"  {class_name}: {params}")
            
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)