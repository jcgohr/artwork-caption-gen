import argparse
import json
from typing import List, Dict

class FusionParser:
    def __init__(self):
        self.runs: List[Dict] = []
        self.output_path: str = ""
        self._parse_arguments()

    def _parse_arguments(self) -> None:
        """Parse command line arguments for run files."""
        parser = argparse.ArgumentParser(
            description='Parse run files for fusion'
        )
        
        # Add output argument
        parser.add_argument(
            '-o',
            '--output',
            required=True,
            help='Path to the output JSON file'
        )
        
        # Add positional arguments for run files
        parser.add_argument(
            'run_files',
            nargs='+',
            help='One or more run JSON files'
        )
        
        # Parse arguments
        args = parser.parse_args()
        
        # Store output path
        self.output_path = args.output
            
        # Load run files
        self.runs = []
        for run_file in args.run_files:
            with open(run_file, 'r') as f:
                self.runs.append(json.load(f))

if __name__ == '__main__':
    # Example usage
    parser = FusionParser()
    print(f"Loaded {len(parser.runs)} run files")
    print(f"Output will be written to: {parser.output_path}")