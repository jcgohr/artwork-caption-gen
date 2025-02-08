import argparse
import json
from pathlib import Path
from typing import Dict, Any

class CaptionMetricParser:
    """
    A parser for handling caption files.
    
    This class provides functionality to parse command line arguments
    for caption and image-caption data JSON files, with built-in validation.
    The loaded data is stored in the captions and data attributes.
    """
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Process caption files",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._setup_arguments()
        self.captions: Dict[str, Any] = {}
        self.data: Dict[str, Any] = {}
    
    def _setup_arguments(self) -> None:
        """Set up the command line arguments."""
        self.parser.add_argument(
            "-c", "--caps",
            type=str,
            required=True,
            help="Path to the captions JSON file"
        )
        self.parser.add_argument(
            "-d", "--data",
            type=str,
            required=True,
            help="Path to the image and caption data JSON file"
        )
    
    def _validate_file(self, file_path: str) -> None:
        """
        Validate that the file exists and is a JSON file.
        
        Args:
            file_path: Path to the file to validate
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a JSON file
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() != '.json':
            raise ValueError(f"File must be a JSON file: {file_path}")
    
    def parse_args(self) -> None:
        """
        Parse and validate the command line arguments, then load the JSON files.
        
        Raises:
            FileNotFoundError: If either file doesn't exist
            ValueError: If either file is not a JSON file
            JSONDecodeError: If either file contains invalid JSON
        """
        args = self.parser.parse_args()
        
        # Validate both files
        self._validate_file(args.caps)
        self._validate_file(args.data)
        
        # Load the JSON files
        with open(args.caps, 'r',encoding='utf-8') as caps_file:
            self.captions = json.load(caps_file)
        
        with open(args.data, 'r',encoding='utf-8') as data_file:
            self.data = json.load(data_file)


# Example usage
if __name__ == "__main__":
    try:
        parser = CaptionMetricParser()
        parser.parse_args()
        print(f"Successfully loaded both JSON files")
        print(f"Captions data keys: {list(parser.captions.keys())}")
        print(f"Image and caption data keys: {list(parser.data.keys())}")
    except Exception as e:
        print(f"Error: {str(e)}")