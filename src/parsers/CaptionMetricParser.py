import argparse
from pathlib import Path
import sys
from utils.mutate import finetune_dataset_format


class CaptionMetricParser:
    """
    A parser for handling caption files.
    
    This class provides functionality to parse command line arguments
    for caption and metadata files, processing them into a consistent
    format for finetuning.
    """
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Process caption files",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._setup_arguments()
        self.key: str = ""
        self.output_path: str = ""
        self.data = None
    
    def _setup_arguments(self) -> None:
        """Set up the command line arguments."""
        self.parser.add_argument(
            "-c", "--caps",
            type=str,
            required=True,
            help="Path to the generated captions JSON file"
        )
        self.parser.add_argument(
            "-d", "--data",
            type=str,
            required=True,
            help="Path to the metadata JSON file"
        )
        self.parser.add_argument(
            "-k", "--key",
            type=str,
            required=True,
            help="Key string to use"
        )
        self.parser.add_argument(
            "-o", "--out",
            type=str,
            required=True,
            help="Output filepath"
        )
    
    def _validate_input_file(self, file_path: str) -> None:
        """
        Validate input file existence.
        
        Args:
            file_path: Path to the input file to validate
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def _ensure_output_directory(self, file_path: str) -> None:
        """
        Ensure the output directory exists, creating it if necessary.
        
        Args:
            file_path: Path to the output file
        """
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def _check_output_file(self, file_path: str) -> bool:
        """
        Check if output file exists and prompt user for overwrite permission.
        
        Args:
            file_path: Path to check
            
        Returns:
            bool: True if file doesn't exist or user agrees to overwrite
        """
        if Path(file_path).exists():
            while True:
                response = input(f"File {file_path} already exists. Overwrite? (y/n): ").lower()
                if response in ('y', 'yes'):
                    return True
                if response in ('n', 'no'):
                    print("Operation cancelled by user")
                    return False
                print("Please answer 'y' or 'n'")
        return True
    
    def parse_args(self) -> None:
        """
        Parse and validate the command line arguments, then process the files.
        
        Raises:
            FileNotFoundError: If input files don't exist
            SystemExit: If user chooses not to overwrite existing output file
        """
        args = self.parser.parse_args()
        
        # Validate input files exist
        self._validate_input_file(args.caps)
        self._validate_input_file(args.data)
        
        # Create output directory if needed and check for file overwrite
        self._ensure_output_directory(args.out)
        if not self._check_output_file(args.out):
            sys.exit(0)
        
        # Store arguments
        self.key = args.key
        self.output_path = args.out
        
        # Process the input files using finetune_dataset_format
        self.data = finetune_dataset_format(
            metadata_path=args.data,
            generated_cap_path=args.caps,
            output_path=None
        )


# Example usage
if __name__ == "__main__":
    try:
        parser = CaptionMetricParser()
        parser.parse_args()
        print(f"Successfully processed input files")
        print(f"Using key: {parser.key}")
        print(f"Output will be written to: {parser.output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)