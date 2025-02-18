import argparse
import json
from typing import List, Dict

class SignificanceTestingParser:
    def __init__(self):
        self.qrel: Dict = {}
        self.runs: List[Dict] = []
        self._parse_arguments()

    def _parse_arguments(self) -> None:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='Parse qrel and run files for significance testing'
        )
        
        # Add the qrel argument
        parser.add_argument(
            '-q', 
            '--qrel',
            required=True,
            help='Path to the qrel JSON file'
        )
        
        # Add positional arguments for run files
        parser.add_argument(
            'run_files',
            nargs='+',
            help='One or more run JSON files'
        )
        
        # Parse arguments
        args = parser.parse_args()
        
        # Load qrel file
        with open(args.qrel, 'r') as f:
            self.qrel = json.load(f)
            
        # Load run files
        self.runs = []
        for run_file in args.run_files:
            with open(run_file, 'r') as f:
                self.runs.append(json.load(f))

if __name__ == '__main__':
    # Example usage
    parser = SignificanceTestingParser()
    print(f"Loaded qrel file with {len(parser.qrel)} entries")
    print(f"Loaded {len(parser.runs)} run files")